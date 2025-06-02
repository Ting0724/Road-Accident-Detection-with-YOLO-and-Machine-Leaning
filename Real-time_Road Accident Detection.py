# Import libraries
from ultralytics import YOLO  # Import YOLO model from Ultralytics
import cv2                   # Import OpenCV library
import math                  # Import math module for mathematical operations
import time                  # Import time module for measuring processing time
import tkinter as tk         # Import tkinter for GUI components
from tkinter import filedialog  # Import filedialog for file selection
import torch
import os
import urllib.request
import numpy as np
import time
import socket

def run_client():
    # create a socket object
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_ip = "192.168.166.19"  # replace with the server's IP address
    server_port = 6000  # replace with the server's port number
    # establish connection with server
    client.connect((server_ip, server_port))
    # input message and send it to the server
    print('Sending data')
    msg = '1'
    client.send(msg.encode("utf-8")[:1024])
    time.sleep(0.1)
    msg = '0'
    client.send(msg.encode("utf-8")[:1024])


def check_gpu():
    if torch.cuda.is_available():
        print("CUDA is available. GPU can be used.")
        print("Number of GPUs available:", torch.cuda.device_count())
        print("GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU.")

check_gpu()

# Replace the URL with the IP camera's stream URL
url = 'http://192.168.166.234/cam-hi.jpg'
cv2.namedWindow("live Cam", cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(url)
 
if not cap.isOpened():
    print("Failed to open the IP camera stream")
    exit()

# Load the YOLO model

model = YOLO("C:/Users/Asus/Downloads/YOLO Result/Trained YOLOv8m-40epochs.pt")  # Load YOLOv8 model with pre-trained weights

print("Load Model")
# Define object classes for detection
classNames = ["ModerateAccident", "NoAccident", "SevereAccident"]


while True:
    print("Start")
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    img = cv2.imdecode(imgnp,-1)

    # Resize the image to 640x480 pixels to match the webcam resolution
    img = cv2.resize(img, (640, 480))

    input_directory = "C:/Users/Asus/Downloads/YOLO Result/Real-time image detection"
    save_inputpath = os.path.join(input_directory, "input.jpg")
    cv2.imwrite(save_inputpath, img)
    print(f"Processed input image saved at {save_inputpath}")

    # Start time
    start_time = time.time()

    # Perform object detection using the YOLO model on the captured frame
    results = model(img, stream=True)

    # End time
    end_time = time.time()

    # Calculate and print the processing time
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.4f} seconds")

    # Iterate through the results of object detection
    for r in results:
        boxes = r.boxes  # Extract bounding boxes for detected objects

        # Iterate through each bounding box
        for box in boxes:
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integer values

            # Draw the bounding box on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Calculate and print the confidence score of the detection
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Determine and print the class name of the detected object
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Draw text indicating the class name on the frame
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            # Check if the detected class is ModerateAccident or SevereAccident
            if classNames[cls] in ["ModerateAccident", "SevereAccident"]:
                print("Call hospital")
                run_client()
                output_directory = "C:/Users/Asus/Downloads/YOLO Result/Real-time image detection"
                save_path = os.path.join(output_directory, "detected.jpg")
                cv2.imwrite(save_path, img)
                print(f"Processed image saved at {save_path}")

    
    # Display the frame with detected objects in a window named "Selected Image"
    cv2.imshow('Live Cam', img)

    if cv2.waitKey(1) == ord('q'):
        break

else:
    print("No file selected")

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
