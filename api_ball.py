import requests
import json
import threading
import cv2
import base64

def cv2_image_to_base64(cv_image):
    # Convert the OpenCV image (BGR) to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Encode the image to a byte array (PNG format in this case)
    _, buffer = cv2.imencode('.png', rgb_image)
    
    # Convert the buffer to a Base64 string
    base64_string = base64.b64encode(buffer).decode('utf-8')
    
    return base64_string

def send_lap_request_in_background(race_id, marble_id, lap, time):
    # pass
    threading.Thread(
        target=send_lap_post_request,
        args=(race_id, marble_id, lap, time),
        daemon=True  # Background thread that won't block on exit
    ).start()

def send_finalorder_request_in_background(race_id, final_order, snapshot):
    # pass
    threading.Thread(
        target=send_racefinal_post_request,
        args=(race_id, final_order, snapshot),
        daemon=True  # Background thread that won't block on exit
    ).start()
    
def send_lap_post_request(race_id, marble_id, lap, time):
    # URL of the API endpoint
    url = "https://p1su5ofsta.execute-api.me-south-1.amazonaws.com/dev?action=raceLive"
    
    # Payload data
    payload = {
        "race_id": race_id,
        "marble_id": marble_id,
        "lap": lap,
        "time": time
    }
    
    # Set headers (optional; include if required by your API)
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send the POST request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            print(f"Request: {race_id, marble_id, lap, time} successful!")
            print(f"{response.text}")
            # return r  # Returns the JSON response
        else:
            print(f"Request: {race_id, marble_id, lap, time} failed with status code:", response.status_code)
            print("Error:", response.text)
            # return None  # Return None in case of failure
    except requests.RequestException as e:
        print("An error occurred:", e)
        # return None

def send_racefinal_post_request(race_id, final_order, snapshot):
    # URL of the API endpoint
    url = "https://p1su5ofsta.execute-api.me-south-1.amazonaws.com/dev?action=raceFinal"
    base64_str = cv2_image_to_base64(snapshot)
    # Payload data
    payload = {
                "race_id": race_id,
                "first_place": int(final_order[0]),
                "second_place": int(final_order[1]),
                "third_place": int(final_order[2]),
                "forth_place": int(final_order[3]),
                "fifth_place": int(final_order[4]),
                "sixth_place": int(final_order[5]),
                "seventh_place": int(final_order[6]),
                "eighth_place": int(final_order[7]),
                "nineth_place": 0,
                "tenth_place": 0,
                "snapshot": "data:image/png;base64,"+base64_str
            }
    # Set headers (optional; include if required by your API)
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send the POST request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            print(f"Request: {race_id, final_order} successful!")
            print(f"{response.text}")
            # return r  # Returns the JSON response
        else:
            print(f"Request: {race_id, final_order} failed with status code:", response.status_code)
            print("Error:", response.text)
            # return None  # Return None in case of failure
    except requests.RequestException as e:
        print("An error occurred:", e)
        # return None


# Call the function and store the result if needed
# result = send_lap_post_request(1, 1, 1, 40)
# print("Response:", result)

