import cv2
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
import pytz
from email.mime.image import MIMEImage
import matplotlib.pyplot as plt
import io
import sqlite3
import time
import json
import numpy as np
import sys


def find_available_camera():
    """
    Attempts to find an available camera by iterating through possible indices.
    
    Returns:
        int: The index of the first available camera, or None if no camera is found.
    """
    # Try camera indices from 0 to 10
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found working camera at index {i}")
            cap.release()
            return i
    print("No available cameras found.")
    return None


def load_config():
    """
    Loads the configuration values from the config file.

    Returns:
        A dictionary containing the configuration values.

    Raises:
        FileNotFoundError: If the config file is not found.
        json.JSONDecodeError: If the config file has invalid JSON syntax.
    """
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config


def video_capture():
    """
    Captures video frames from the camera and performs object detection (YOLO) on each frame.
    """
    config = load_config()
    
    # Find an available camera if the configured index doesn't work
    camera_index = config["video_url"]
    video = cv2.VideoCapture(camera_index)
    
    # If the configured camera doesn't work, try to find an available camera
    if not video.isOpened():
        print(f"Could not open camera at index {camera_index}")
        available_camera = find_available_camera()
        
        if available_camera is None:
            print("No cameras are available. Exiting.")
            sys.exit(1)
        
        # Update the configuration and reopen with the new index
        config["video_url"] = available_camera
        video = cv2.VideoCapture(available_camera)

    # Set camera properties for better performance
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = video.read()

            if not ret:
                print("Failed to capture frame. Reconnecting...")
                video.release()
                video = cv2.VideoCapture(config["video_url"])
                continue

            # Perform object detection
            detected_objects = model_predict(frame)

            for detected_object in detected_objects:
                class_name = detected_object['class']
                coordinates = detected_object['coordinates']
                probability = detected_object['probability']

                # Draw bounding box on the frame
                cv2.rectangle(frame, 
                             (coordinates[0], coordinates[1]), 
                             (coordinates[2], coordinates[3]), 
                             (0, 255, 0), 2)
                
                # Put text on the frame
                cv2.putText(frame, 
                           f"{class_name} {probability:.2f}", 
                           (coordinates[0], coordinates[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                           (0, 255, 0), 2)

                if class_name == 'person' and probability > 0.5:
                    # Get current timestamp
                    timestamp = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Trigger email and database logging
                    mail_trigger(frame, timestamp, config)
                    database_entry(frame, timestamp, config)
            
            # Display the frame
            cv2.imshow('Intruder Detection', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Always release resources
        video.release()
        cv2.destroyAllWindows()


def model_predict(frame):
    """
    Performs object detection using a trained YOLO model on the given frame.

    Args:
        frame: The input frame on which object detection will be performed.

    Returns:
        List of dictionaries containing the class name, coordinates, and probability for each detected object.
    """
    config = load_config()
    best_model = YOLO(config["model_weights_path"])
    
    # Perform prediction directly on the frame
    predictions = best_model.predict(frame, conf=0.5)
    detected_objects = []
    
    if len(predictions) > 0:
        prediction = predictions[0]
        for box in prediction.boxes:
            class_name = prediction.names[box.cls[0].item()]
            coordinates = box.xyxy[0].tolist()
            coordinates = [round(x) for x in coordinates]
            probability = round(box.conf[0].item(), 2)
            print(f"Detected: {class_name} with probability {probability}")
            
            # Change condition to detect 'person' instead of 'Intruder'
            if class_name == 'person' and probability > 0.5:
                detected_object = {
                    'class': class_name,
                    'coordinates': coordinates,
                    'probability': probability
                }
                detected_objects.append(detected_object)

    return detected_objects


def mail_trigger(frame, timestamp, config):
    """
    Sends an email alert with an attached image and timestamp information.
    """
    sender_email = config["sender_email"]
    sender_password = config["sender_password"]
    smtp_server = config["smtp_server"]
    smtp_port = config["smtp_port"]
    recipient_email = config["recipient_email"]

    # Format timestamp for email
    timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    timestamp = timestamp.strftime('%d-%B-%Y [%A], %H:%M:%S')

    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = '*Intruder Alert' + ' : ' + timestamp + ' ' + 'Hours*'

    # Convert frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,10))
    plt.imshow(frame_rgb)
    plt.axis('off')

    image_io = io.BytesIO()
    plt.savefig(image_io, format='jpeg', bbox_inches='tight', pad_inches=0)
    image_io.seek(0)
    plt.close()

    # Attach the image
    image = MIMEImage(image_io.getvalue(), _subtype='jpeg')
    image.add_header("Content-Disposition", "attachment", filename="intruder_image.jpg")
    message.attach(image)

    text = MIMEText(
        f"Dear Control Room, \n\nThis is to keep you informed that a person has been detected at {timestamp} Hours")
    message.attach(text)

    # Send the email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print("Failed to send email:", str(e))


def database_entry(frame, timestamp, config):
    """
    Inserts an image and timestamp into a SQLite database table for logging.
    """
    connection = sqlite3.connect(config["database_path"])
    cursor = connection.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB,
            captured_time TEXT
        )
    ''')

    # Encode image
    _, image_bytes = cv2.imencode('.jpg', frame)

    # Insert data
    sql = """INSERT INTO detection_log (image, captured_time) VALUES (?, ?)"""
    cursor.execute(sql, (image_bytes.tobytes(), timestamp))
    connection.commit()

    cursor.close()
    connection.close()


def main():
    """
    Main entry point of the script with error handling.
    """
    try:
        video_capture()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()