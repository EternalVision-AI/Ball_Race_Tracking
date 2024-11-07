import cv2
import os
import numpy as np
import math
from collections import Counter
from datetime import datetime
# Constants.
INPUT_WIDTH = 320
INPUT_HEIGHT = 320

recognition_classes = ['White', 'Black', 'Blue', 'Green', 'Yellow', 'Orange', 'Red', 'Purple']


confThreshold = 0.6  # Confidence threshold
nmsThreshold = 0.6 # Non-maximum suppression threshold
dir_path = os.path.dirname(os.path.realpath(__file__))
detection_model = cv2.dnn.readNetFromONNX(dir_path + "/ballv8n.onnx")


# Define BGR color values for each class
class_colors = {
    'White': (255, 255, 255),
    'Black': (0, 0, 0),
    'Blue': (255, 0, 0),
    'Green': (0, 255, 0),
    'Yellow': (0, 255, 255),
    'Orange': (0, 165, 255),
    'Red': (0, 0, 255),
    'Purple': (255, 0, 255)
}

class_points = {
    'White': [1, 1],
    'Black': [1, 1],
    'Blue': [1, 1],
    'Green': [1, 1],
    'Yellow': [1, 1],
    'Orange': [1, 1],
    'Red': [1, 1],
    'Purple': [1, 1]
}

class_points_prev = {
    'White': [1, 1],
    'Black': [1, 1],
    'Blue': [1, 1],
    'Green': [1, 1],
    'Yellow': [1, 1],
    'Orange': [1, 1],
    'Red': [1, 1],
    'Purple': [1, 1]
}

class_race = {
    'White': [0, 0, 0],
    'Black': [0, 0, 0],
    'Blue': [0, 0, 0],
    'Green': [0, 0, 0],
    'Yellow': [0, 0, 0],
    'Orange': [0, 0, 0],
    'Red': [0, 0, 0],
    'Purple': [0, 0, 0],
}

class_race_time = {
    'White': 0,
    'Black': 0,
    'Blue': 0,
    'Green': 0,
    'Yellow': 0,
    'Orange': 0,
    'Red': 0,
    'Purple': 0,
}

class_whole_time = {
    'White': 0,
    'Black': 0,
    'Blue': 0,
    'Green': 0,
    'Yellow': 0,
    'Orange': 0,
    'Red': 0,
    'Purple': 0,
}


def DetectionProcess(original_image):
	height, width, _ = original_image.shape
	length = max((height, width))
	image = np.zeros((length, length, 3), np.uint8)
	image[0:height, 0:width] = original_image
	scale = length / INPUT_WIDTH

	blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(INPUT_WIDTH, INPUT_WIDTH), swapRB=True)
	detection_model.setInput(blob)
	outputs = detection_model.forward()

	outputs = np.array([cv2.transpose(outputs[0])])
	rows = outputs.shape[1]

	boxes = []
	scores = []
	class_ids = []

	for i in range(rows):
		classes_scores = outputs[0][i][4:]
		(minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
		if maxScore >= confThreshold:
			box = [
				outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
				outputs[0][i][2], outputs[0][i][3]]
			boxes.append(box)
			scores.append(maxScore)
			class_ids.append(maxClassIndex)

	result_boxes = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold)

	detections = []
	for i in range(len(result_boxes)):
		index = result_boxes[i]
		box = boxes[index]
		detection = {
			'class_id': class_ids[index],
			'class_name': recognition_classes[class_ids[index]],
			'confidence': scores[index],
			'box': box,
			'scale': scale}
		detections.append(detection)
	return detections

def draw_class_rectangle(img, left, top, right, bottom, class_name):
    # Get the color for the class from the dictionary
    color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class not found
    # Draw the rectangle
    # cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    
    # Calculate the center of the rectangle
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    
    # Calculate the radius as half the width or height (whichever is smaller)
    radius = min((right - left) // 2, (bottom - top) // 2)
    
    # Draw the circle on the image
    cv2.circle(img, (center_x, center_y), radius, color, 2)
    return img

def draw_startline(img):
		start_point = (0, 150)
		end_point = (768, 150)
		color = (0, 0, 255)  # Green
		thickness = 5
		cv2.line(img, start_point, end_point, color, thickness)
		return img

def rotate_point_clockwise(x, y, angle_degrees = -6.230828025477707006369426751592):
    # Convert the angle from degrees to radians
    angle_radians = math.radians(angle_degrees)
    
    # Calculate the new x and y coordinates using the rotation formula
    new_x = x * math.cos(angle_radians) + y * math.sin(angle_radians)
    new_y = -x * math.sin(angle_radians) + y * math.cos(angle_radians)
    
    return new_x, new_y-180

isdisplayed = False
def DetectCard(img, timestamp_sec):
	global isdisplayed
	global class_points, class_points_prev, class_race, class_race_time, class_whole_time
	detections = DetectionProcess(img)
	detected_cards = []
	for detection in detections:
		class_id, class_name, confidence, box, scale = \
			detection['class_id'], detection['class_name'], detection['confidence'], detection['box'], detection[
				'scale']
		left, top, right, bottom = round(box[0] * scale), round(box[1] * scale), round(
			(box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)
		# Ensure bounding box is within image dimensions
		left = max(left, 0)
		top = max(top, 0)
		right = min(right, img.shape[1] - 1)
		bottom = min(bottom, img.shape[0] - 1)
    
		detected_cards.append([left, top, right, bottom])
  
		
		new_x0, new_y0 = rotate_point_clockwise(600-512, 550)
		new_x, new_y = rotate_point_clockwise(left, top)
		img = draw_class_rectangle(img, left, top, right, bottom, class_name)
		# cv2.putText(img, closest_color_class, (left, top), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
		cv2.putText(img, str(round(new_x))+", "+str(round(new_y)), (left, top), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
		# cv2.putText(img, str(round(new_x0))+", "+str(round(new_y0)), (int(new_x0), int(new_y0)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

		class_points[class_name] = (new_x, new_y)
		if class_points_prev[class_name][1] > 0 and class_points[class_name][1] < 0:
			# print("Pass", class_name)
			# Record end time
			end_time = timestamp_sec
			# Calculate elapsed time

			class_race[class_name][0] += 1
			class_race[class_name][2] = end_time - class_race[class_name][1]
			class_race[class_name][1] = end_time
			if class_race[class_name][0] > 0 and len(detections) != 8:
				# print(str(class_race[class_name][2]))
				if class_race[class_name][2] > 10 and class_race[class_name][2] <= 100:
					class_whole_time[class_name] += class_race[class_name][2]
					print(class_name + " in Lap "+str(class_race[class_name][0]) + " = "+str(class_race[class_name][2]))
					class_race[class_name][2] = 0
				else:
					class_race[class_name][0] -= 1
			if class_race[class_name][0] > 0 and len(detections) == 8 and class_points[class_name][1] > -60:
				
				if class_race[class_name][2] > 10 and class_race[class_name][2] <= 100:
					class_whole_time[class_name] += class_race[class_name][2]
					print(class_name + " in Lap "+str(class_race[class_name][0]) + " = "+str(class_race[class_name][2]))
					class_race[class_name][2] = 0
				else:
					class_race[class_name][0] -= 1
		if class_race[class_name][0] > 0 and class_points[class_name][1] > 0 and ((class_points_prev[class_name][1] > 0 and class_points[class_name][1] > 0 and class_points_prev[class_name][1] -  class_points[class_name][1] < 2 and class_points_prev[class_name][1] -  class_points[class_name][1]>0) or len(detections) == 8):
			# Record end time
			end_time = timestamp_sec

			if class_race_time[class_name] == 0:
				class_race[class_name][0] += 1
				class_race[class_name][2] = end_time - class_race[class_name][1]
				class_race[class_name][1] = end_time
				class_race_time[class_name] = end_time
    
				class_whole_time[class_name] += class_race[class_name][2]
				# print(class_name + " in Lap final "+str(class_race[class_name][0]) + " = "+str(class_race[class_name][2]))


		class_points_prev[class_name] = (new_x, new_y)
  
  # Check if all y coordinates in class_points are greater than 0
	all_y_positive = all(point[1] > 0 for point in class_points.values())
	all_y_positive_prev = all(point[1] > 0 for point in class_points_prev.values())
	# print(all_y_positive)
	if not all_y_positive:
		isdisplayed = False
	if all_y_positive and isdisplayed is False and len(detections) == 8:
			
			# Filter and sort class points based on y-coordinate
			sorted_classes = [(name, point[1]) for name, point in class_points.items() if point[1] > 0]
			sorted_classes.sort(key=lambda x: x[1])  # Sort by y-coordinate
   
			for idx, (class_name, y_coord) in enumerate(sorted_classes):
					if idx == 7 and class_race[class_name][0] != 0:
						end_time = timestamp_sec
						class_race[class_name][0] += 1
						class_race[class_name][2] = end_time - class_race[class_name][1]
						class_whole_time[class_name] += class_race[class_name][2]
						if class_race_time[class_name] == 0:
							class_race_time[class_name] = class_race[class_name][1]
					print(f"{class_name} = {str(class_whole_time[class_name])}")
			for idx, (class_name, y_coord) in enumerate(sorted_classes):
					print(f"p{idx+1} : {class_name}")
     
			class_points = {
					'White': [0, 0],
					'Black': [0, 0],
					'Blue': [0, 0],
					'Green': [0, 0],
					'Yellow': [0, 0],
					'Orange': [0, 0],
					'Red': [0, 0],
					'Purple': [0, 0]
			}
			class_points_prev = {
					'White': [0, 0],
					'Black': [0, 0],
					'Blue': [0, 0],
					'Green': [0, 0],
					'Yellow': [0, 0],
					'Orange': [0, 0],
					'Red': [0, 0],
					'Purple': [0, 0]
			}
			class_race = {
					'White': [0, 0, 0],
					'Black': [0, 0, 0],
					'Blue': [0, 0, 0],
					'Green': [0, 0, 0],
					'Yellow': [0, 0, 0],
					'Orange': [0, 0, 0],
					'Red': [0, 0, 0],
					'Purple': [0, 0, 0],
			}
			class_race_time = {
					'White': 0,
					'Black': 0,
					'Blue': 0,
					'Green': 0,
					'Yellow': 0,
					'Orange': 0,
					'Red': 0,
					'Purple': 0,
			}
			class_whole_time = {
					'White': 0,
					'Black': 0,
					'Blue': 0,
					'Green': 0,
					'Yellow': 0,
					'Orange': 0,
					'Red': 0,
					'Purple': 0,
			}
     
			isdisplayed = True
  

  

	img = draw_startline(img)
	# Resize the image to half of its original dimensions
	img_resized = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))

	# Display the resized image
	cv2.imshow("Race", img)
	cv2.waitKey(1)
	return img

def process_video(video_path):
    # Open the video file
		# cap = cv2.VideoCapture(video_path)
		cap = cv2.VideoCapture(0)
		desired_fps = 60
		cap.set(cv2.CAP_PROP_FPS, desired_fps)
		fps = cap.get(cv2.CAP_PROP_FPS)
		print(f"Camera FPS: {fps}")
    # Check if video file opened successfully
		if not cap.isOpened():
				print(f"Error opening video file: {video_path}")
				return

		frame_count = 0
    # Loop over each frame in the video
				
		while True:
				ret, frame = cap.read()
				if not ret:
						break  # Stop if no more frames are returned

				 # Get the current frame timestamp in milliseconds
				timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Convert to seconds (optional)
				timestamp_sec = timestamp_ms / 1000.0
				# Define the codec and create VideoWriter object to write the video
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 video
				# output_video = cv2.VideoWriter('output_video.mp4', fourcc, 30, (int(frame.shape[1] * 2/5), frame.shape[0]))  # 30 fps, half-sized frame
				if frame_count % 2 == 0:
					# Get frame dimensions
					height, width, _ = frame.shape
					# print(height, width)
					# print(str(width*2/5)) #512
					
					# Cut the middle portion (width * 2/5 to width * 4/5)
					start_x = int(width * 1 / 5)
					end_x = int(width * 4 / 5)
					
					# Slice the frame horizontally to keep only the middle portion
					middle_frame = frame[:, start_x:end_x]
					# img = None
					# Darken the image by multiplying it with a factor less than 1
					dark_factor = 1  # Adjust this value to control darkness level (0.0 to 1.0)
					bright_factor = 0
					img_darkened = cv2.convertScaleAbs(middle_frame, alpha=dark_factor, beta=bright_factor)
					# Pass the cropped frame to the DetectCard function
					img = DetectCard(middle_frame, timestamp_sec)
					# Write the resized frame to the video file
					# output_video.write(img)
				frame_count += 1

    # Release the video capture object
		# cap.release()

if __name__ == '__main__':
    video_path = "./vid/(1).mp4"  # Change to your video path
    # Record start time
    
    process_video(video_path)






