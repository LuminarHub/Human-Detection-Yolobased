from ultralytics import YOLO
import cv2

# Load the YOLO model (use a pre-trained model for object detection)
model = YOLO('yolov8s.pt')  # 'yolov8s.pt' is a small model with better accuracy compared to 'yolov8n.pt'.

# Open the webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or replace with the camera index if multiple cameras are present.

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        break

    # Perform detection on the current frame
    results = model(frame)

    # Get detection results
    human_count = 0
    for result in results[0].boxes:
        class_id = int(result.cls)  # Get the class ID of the detected object
        confidence = result.conf.item()  # Convert tensor to a Python float
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Extract bounding box coordinates

        if class_id == 0:  # Class 0 corresponds to 'person' in the COCO dataset
            human_count += 1
            # Draw bounding box and label for humans
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Human ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0q.6, (255, 0, 0), 2)

    # Display human count on the frame
    cv2.putText(frame, f'Humans detected: {human_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Print human count to the console
    print(f"Humans detected: {human_count}")

    # Show the frame with detection results
    cv2.imshow("Human Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()