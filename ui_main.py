import cv2
import os
import numpy as np
import math
from collections import Counter
from datetime import datetime
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog
from PIL import Image, ImageTk
from api_ball import send_request_in_background
import threading

# Constants.
INPUT_WIDTH = 320
INPUT_HEIGHT = 320

recognition_classes = ['White', 'Black', 'Blue', 'Green', 'Yellow', 'Orange', 'Red', 'Purple']


confThreshold = 0.6  # Confidence threshold
nmsThreshold = 0.7 # Non-maximum suppression threshold
dir_path = os.path.dirname(os.path.realpath(__file__))
detection_model = cv2.dnn.readNetFromONNX(dir_path + "/model/ballv8n.onnx")

# Initial parameter
race_id = 1
startline_y = 150
detection_fps = 4
isdisplayed = False


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
		global startline_y
		start_point = (0, startline_y)
		end_point = (img.shape[1], startline_y)
		color = (0, 255, 0)  # Green
		thickness = 2
		cv2.line(img, start_point, end_point, color, thickness)
		return img

def rotate_point_clockwise(x, y, angle_degrees = -6.230828025477707006369426751592):
    # Convert the angle from degrees to radians
    angle_radians = math.radians(angle_degrees)
    
    # Calculate the new x and y coordinates using the rotation formula
    new_x = x * math.cos(angle_radians) + y * math.sin(angle_radians)
    new_y = -x * math.sin(angle_radians) + y * math.cos(angle_radians)
    
    return new_x, new_y-startline_y-30


def DetectCard(img, timestamp_sec):
	global isdisplayed, race_id
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
					ball_index = recognition_classes.index(class_name) + 1
					send_request_in_background(race_id=race_id, marble_id=ball_index, lap=class_race[class_name][0], time=class_race[class_name][2])
					class_race[class_name][2] = 0
				else:
					class_race[class_name][0] -= 1
			if class_race[class_name][0] > 0 and len(detections) == 8 and class_points[class_name][1] > -60:
				
				if class_race[class_name][2] > 10 and class_race[class_name][2] <= 100:
					class_whole_time[class_name] += class_race[class_name][2]
					print(class_name + " in Lap "+str(class_race[class_name][0]) + " = "+str(class_race[class_name][2]))
					ball_index = recognition_classes.index(class_name) + 1
					send_request_in_background(race_id=race_id, marble_id=ball_index, lap=class_race[class_name][0], time=class_race[class_name][2])
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
					if class_whole_time[class_name] != 0:
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
	# cv2.imshow("Race", img)
	# cv2.waitKey(1)
	return img





# Set the appearance mode to 'dark' to use the dark theme globally
ctk.set_appearance_mode("dark")  # Options are 'dark', 'light', or 'system' for automatic

class App(ctk.CTk):
		def __init__(self):
				super().__init__()

				self.title("Race Detection Management")
				# Maximize the window to fill the screen
				self.attributes("-fullscreen", True)  # Enable full screen mode
				

				# Configure grid for the main window
				self.grid_columnconfigure(0, weight=1)
				self.grid_columnconfigure(1, weight=2)
				self.grid_columnconfigure(2, weight=4)
				# self.grid_rowconfigure(0, weight=1)
				# self.grid_rowconfigure(1, weight=0)
				# Create a left frame
				self.left_frame = ctk.CTkFrame(self)
				self.left_frame.grid(row=0, column=0, padx=(20, 5), pady=20, sticky="new")
				# Left side - Column 0 widgets
				self.label2 = ctk.CTkLabel(self.left_frame, text="Race Control")
				self.label2.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

				self.input_raceid = ctk.CTkEntry(self.left_frame, placeholder_text=race_id)
				self.input_raceid.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

				self.btn_set_raceid = ctk.CTkButton(self.left_frame, text="Set the Race ID", command=self.set_raceid)
				self.btn_set_raceid.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

				self.label_status = ctk.CTkLabel(self.left_frame, text="Race Status")
				self.label_status.grid(row=3, column=0, padx=10, pady=(10, 2), sticky="w")
    
				self.console_box = ctk.CTkTextbox(self.left_frame, height=250)
				self.console_box.grid(row=4, column=0, padx=10, pady=(2, 15), sticky="ew")
				self.console_box.configure(state="disabled")

				# Additional widgets in the left frame
				# Add a dropdown menu for race status
				

				# self.dropdown_status = ctk.CTkOptionMenu(self.left_frame, values=["Scheduled", "Ongoing", "Completed", "Cancelled"])
				# self.dropdown_status.grid(row=5, column=0, padx=10, pady=2, sticky="ew")

				# # Add a checkbox for race notifications
				# self.checkbox_notifications = ctk.CTkCheckBox(self.left_frame, text="Enable Notifications")
				# self.checkbox_notifications.grid(row=6, column=0, padx=10, pady=(15, 2), sticky="ew")

				# Add a button for saving settings
				self.btn_reset = ctk.CTkButton(self.left_frame, text="Reset all races", command=self.reset_allraces)
				self.btn_reset.grid(row=7, column=0, padx=10, pady=10, sticky="ew")
    
				# Add a button for saving settings
				self.btn_connect = ctk.CTkButton(self.left_frame, text="Connect the camera", command=self.process_video)
				self.btn_connect.grid(row=8, column=0, padx=10, pady=10, sticky="ew")

				# Configure the left_frame to expand with the window
				self.left_frame.grid_rowconfigure(8, weight=1)  # Allows the last row to expand
				self.left_frame.grid_columnconfigure(0, weight=1)  # Ensures all widgets expand horizontally

    
				# Create a right frame with two columns
				self.right_frame = ctk.CTkFrame(self)
				self.right_frame.grid(row=0, column=1, padx=(5, 5), pady=20, sticky="nsew")
				self.right_frame.grid_columnconfigure(0, weight=1)  # First column
				self.right_frame.grid_columnconfigure(1, weight=1)  # Second column

				# Left side - Column 0 widgets
				self.label1 = ctk.CTkLabel(self.right_frame, text="Race Management")
				self.label1.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

				# Configure grid for the main window
				self.grid_columnconfigure(0, weight=1)
				self.grid_rowconfigure(0, weight=1)
				# Create a main frame with two columns
				self.middle_frame = ctk.CTkFrame(self)
				self.middle_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
				self.middle_frame.grid_columnconfigure(0, weight=1)  # First column
				self.middle_frame.grid_columnconfigure(1, weight=1)  # Second column

				# Left side - Column 0 widgets
				self.label2 = ctk.CTkLabel(self.middle_frame, text="Fine-tuning")
				self.label2.grid(row=0, column=0, padx=10, pady=10, sticky="e")

				self.input_startline_y = ctk.CTkEntry(self.middle_frame, placeholder_text=startline_y)
				self.input_startline_y.grid(row=1, column=0, padx=10, pady=2, sticky="ew")


				self.btn_set_startline_y = ctk.CTkButton(self.middle_frame, text="Set the Start Line Y", command=self.set_startline_y)
				self.btn_set_startline_y.grid(row=2, column=0, padx=10, pady=2, sticky="ew")

				# Right side - Column 1 widgets
				self.label3 = ctk.CTkLabel(self.middle_frame, text="Model")
				self.label3.grid(row=0, column=1, padx=10, pady=10, sticky="w")

				self.input_fps = ctk.CTkEntry(self.middle_frame, placeholder_text=detection_fps)
				self.input_fps.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

				self.btn_setfps = ctk.CTkButton(self.middle_frame, text="Set the Detection fps", command=self.set_fps)
				self.btn_setfps.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
    
				self.input_confscore = ctk.CTkEntry(self.middle_frame, placeholder_text=confThreshold)
				self.input_confscore.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

				self.btn_confscore = ctk.CTkButton(self.middle_frame, text="Set the Detection Conf.Score", command=self.set_confscore)
				self.btn_confscore.grid(row=4, column=1, padx=10, pady=5, sticky="ew")
    
				# Right side - Column 1 widgets
				self.label3 = ctk.CTkLabel(self.middle_frame, text="Model")
				self.label3.grid(row=0, column=1, padx=10, pady=10, sticky="w")

				self.input_fps = ctk.CTkEntry(self.middle_frame, placeholder_text=detection_fps)
				self.input_fps.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

				self.btn_setfps = ctk.CTkButton(self.middle_frame, text="Set the Detection fps", command=self.set_fps)
				self.btn_setfps.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
    
				self.input_confscore = ctk.CTkEntry(self.middle_frame, placeholder_text=confThreshold)
				self.input_confscore.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

				self.btn_confscore = ctk.CTkButton(self.middle_frame, text="Set the Detection Conf.Score", command=self.set_confscore)
				self.btn_confscore.grid(row=4, column=1, padx=10, pady=5, sticky="ew")
    
				# Create a main frame with two columns
    		# Calculate frame width as a percentage of screen width (e.g., 30%)
				# Set the window to maximum size
				screen_width = self.winfo_screenwidth()
				screen_height = self.winfo_screenheight()
				# frame_width = int(screen_width * 0.6)
				# Right frame setup without a hardcoded width
				self.right_frame = ctk.CTkFrame(self)
				self.right_frame.grid(row=0, column=2, padx=(5, 20), pady=20, sticky="nsew")

				# Configure the right frame grid for expanding its widgets
				self.right_frame.grid_columnconfigure(0, weight=1)  # Expands horizontally
				self.right_frame.grid_rowconfigure(0, weight=1)     # Expands vertically

				self.camera_frame = ctk.CTkLabel(self.right_frame, text="", height=screen_height)
				self.camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

				
				

		def connect_camera(self):
				# process_video()
				self.video_thread = threading.Thread(target=self.process_video)
				self.video_thread.start()
    
		def set_raceid(self):
				global race_id
				try:
						race_id = int(self.input_raceid.get())
						# Update console box with the FPS setting
						self.console_box.configure(state="normal")
						self.console_box.insert("end", f"Race ID set to: {race_id}\n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
				except ValueError:
						self.console_box.configure(state="normal")
						self.console_box.insert("end", "Invalid input for Race ID. Please enter a number.\n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
						self.console_box.configure(state="disabled")
      
		def set_fps(self):
				global detection_fps
				try:
						detection_fps = int(self.input_fps.get()) 
						# Update console box with the FPS setting
						self.console_box.configure(state="normal")
						self.console_box.insert("end", f"Detection FPS set to: {detection_fps}\n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
				except ValueError:
						self.console_box.configure(state="normal")
						self.console_box.insert("end", "Invalid input for FPS. Please enter a number.\n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
						self.console_box.configure(state="disabled")
    
		def set_confscore(self):
				global confThreshold
				try:
						confThreshold = float(self.input_confscore.get())
						# Update console box with the FPS setting
						self.console_box.configure(state="normal")
						self.console_box.insert("end", f"Detection ConfScore set to: {confThreshold}\n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
						self.console_box.configure(state="disabled")
				except ValueError:
						self.console_box.configure(state="normal")
						self.console_box.insert("end", "Invalid input for ConfScore. Please enter a number.\n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
						self.console_box.configure(state="disabled")
    
		def set_startline_y(self):
				global startline_y
				try:
						startline_y = int(self.input_startline_y.get())
						# Update console box with the FPS setting
						self.console_box.configure(state="normal")
						self.console_box.insert("end", f"Start Line Y set to: {startline_y}\n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
						self.console_box.configure(state="disabled")
				except ValueError:
						self.console_box.configure(state="normal")
						self.console_box.insert("end", "Invalid input for Start Line Y. Please enter a number.\n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
						self.console_box.configure(state="disabled")
    
		def reset_allraces(self):
				global isdisplayed, race_id
				global class_points, class_points_prev, class_race, class_race_time, class_whole_time
				try:
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
						race_id = 1
						# Clear any existing text
						self.input_raceid.delete(0, "end")
						# Insert the new value
						self.input_raceid.insert(0, 1)
						# Update console box with the FPS setting
						self.console_box.configure(state="normal")
						self.console_box.insert("end", "All reset! \n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
						self.console_box.configure(state="disabled")
				except ValueError:
						self.console_box.configure(state="normal")
						self.console_box.insert("end", "Reset Failed! \n")
						self.console_box.yview("end")  # Auto-scroll to the latest entry
						self.console_box.configure(state="disabled")

    
		def process_video(self):
				global detection_fps
				# Open the video file
				video_path = "./vid/(1).mp4"  # Change to your video path
				cap = cv2.VideoCapture(video_path)
				# cap = cv2.VideoCapture(0)
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

						if frame_count % detection_fps == 0:
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
							# Display the image in an OpenCV window
							# cv2.imshow("Detected Card", img)
							# Resize frame to label size
							# Resize frame based on label dimensions
							self.camera_frame.update()
							label_width = self.camera_frame.winfo_width()
							label_height = self.camera_frame.winfo_height()
							resized_frame = cv2.resize(img, (label_width, label_height))
							# Convert frame to CTkImage for display
							img = ctk.CTkImage(Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)), size=(label_width, label_height))
							self.camera_frame.configure(image=img)
							self.camera_frame.image = img
							
							# Check for 'q' or 'Esc' key press to exit the loop
							key = cv2.waitKey(1)  # Wait 1 ms between frames for input
							if key == ord('q') or key == 27:  # 27 is the ASCII code for Esc
									break
						 # Schedule the next frame update

				cv2.destroyAllWindows()

# if __name__ == '__main__':
#     video_path = "./vid/(1).mp4"  # Change to your video path
#     # Record start time
#     process_video()

# Create and run the application
if __name__ == "__main__":
		app = App()
		app.mainloop()





