import requests
import json
import threading


def send_request_in_background(race_id, marble_id, lap, time):
    threading.Thread(
        target=send_post_request,
        args=(race_id, marble_id, lap, time),
        daemon=True  # Background thread that won't block on exit
    ).start()
    
def send_post_request(race_id, marble_id, lap, time):
    # URL of the API endpoint
    url = "https://fptljl3i7l.execute-api.me-south-1.amazonaws.com/dev"
    
    # Payload data
    payload = {
        "body": json.dumps({
            "race_id": race_id,
            "marble_id": marble_id,
            "lap": lap,
            "time": time
        })
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

# Call the function and store the result if needed
# result = send_post_request(1, 1, 1, 40)
# print("Response:", result)

