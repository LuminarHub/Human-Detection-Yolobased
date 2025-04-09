from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import cv2
import threading
import numpy as np

app = Flask(__name__, static_folder='static')

class HumanDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Load YOLO model
        self.camera = None
        self.lock = threading.Lock()
        self.current_human_count = 0
        self.max_human_count = 0
        self.frame = None
        self.is_camera_running = False
        self.camera_index = None  # Store the working camera index

    def start_camera(self):
        if self.is_camera_running:
            return False  # Prevent multiple initializations

        print("Attempting to initialize the camera...")

        # Try multiple camera indices to find a working one
        for index in [0, 1, 2, 3, -1]:  # Include multiple possible indices
            self.camera = cv2.VideoCapture(index, cv2.CAP_ANY)  # Try different backends
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Test if the camera works by capturing a frame
            ret, frame = self.camera.read()
            if ret and frame is not None and not np.all(frame == 0):
                print(f"Camera initialized successfully at index {index}")
                self.frame = frame
                self.is_camera_running = True
                self.camera_index = index  # Store the successful index
                return True
            
            # Release if the camera index is not valid
            self.camera.release()

        print("Failed to initialize camera on all indices.")
        self.camera = None
        return False

    def stop_camera(self):
        with self.lock:
            if self.camera:
                self.camera.release()
                self.camera = None
            self.is_camera_running = False
            self.current_human_count = 0
            self.max_human_count = 0
            self.camera_index = None  # Reset the index

            # Show "Camera Stopped" on screen
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.frame, 'Camera Stopped', 
                        (100, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return True

    def detect_humans(self, frame):
        try:
            results = self.model(frame)  # Run YOLO detection
            human_count = 0

            for result in results[0].boxes:
                class_id = int(result.cls)
                confidence = result.conf.item()
                x1, y1, x2, y2 = map(int, result.xyxy[0])

                if class_id == 0:  # Person class
                    human_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'Human ({confidence:.2f})', 
                                (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            with self.lock:
                self.current_human_count = human_count
                self.max_human_count = max(self.max_human_count, human_count)

            cv2.putText(frame, f'Humans detected: {human_count}', 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            return frame
        except Exception as e:
            print(f"Error in human detection: {e}")
            return frame

    def generate_frames(self):
        while self.is_camera_running:
            try:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if not ret or frame is None:
                        print(f"Camera at index {self.camera_index} stopped working.")
                        self.stop_camera()
                        continue  # Skip this frame
                else:
                    frame = self.frame  # Use last known good frame

                processed_frame = self.detect_humans(frame)
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            except Exception as e:
                print(f"Error in frame generation: {e}")
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, 'Camera Error', 
                            (100, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                _, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Create an instance of the HumanDetector
detector = HumanDetector()

@app.route('/')
def index():
    detector.start_camera()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detector.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    with detector.lock:
        return jsonify({
            'current_human_count': detector.current_human_count,
            'max_human_count': detector.max_human_count,
            'is_camera_running': detector.is_camera_running
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
