from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import cv2
import threading
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
import sqlite3
import json

app = Flask(__name__, static_folder='static')

# Load Configurations
with open('config.json') as config_file:
    config = json.load(config_file)

class HumanDetector:
    def __init__(self):
        self.model = YOLO(config["model_weights_path"])  # Load YOLO model
        self.camera = None
        self.lock = threading.Lock()
        self.current_human_count = 0
        self.max_human_count = 0
        self.frame = None
        self.is_camera_running = False
        self.camera_index = None
        self.last_alert_time = None  # To prevent email spam

        # Start YOLO in a separate thread
        self.detection_thread = threading.Thread(target=self.detect_loop, daemon=True)
        self.detection_thread.start()

        # Initialize SQLite database
        self.init_database()

    def start_camera(self):
        if self.is_camera_running:
            return False  # Prevent multiple initializations

        print("Attempting to initialize the camera...")
        for index in [0, 1, 2, 3, -1]:  # Try multiple camera indices
            self.camera = cv2.VideoCapture(index, cv2.CAP_ANY)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            ret, frame = self.camera.read()
            if ret and frame is not None and not np.all(frame == 0):
                print(f"Camera initialized successfully at index {index}")
                self.frame = frame
                self.is_camera_running = True
                self.camera_index = index
                return True
            
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
            self.camera_index = None

            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.frame, 'Camera Stopped', 
                        (100, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return True

    def detect_loop(self):
        while True:
            if self.is_camera_running and self.camera:
                ret, frame = self.camera.read()
                if ret:
                    with self.lock:
                        self.frame = self.detect_humans(frame)

    def detect_humans(self, frame):
        try:
            results = self.model(frame)
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

            # Log detection if a human is found
            if human_count > 0:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.log_detection(timestamp, human_count)
                self.send_alert(timestamp, human_count)

            cv2.putText(frame, f'Humans detected: {human_count}', 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            return frame
        except Exception as e:
            print(f"Error in human detection: {e}")
            return frame

    def generate_frames(self):
        while self.is_camera_running:
            print("📸 Capturing frame...")  # Debugging print
            with self.lock:
                if self.frame is None:
                    print("⚠️ No frame available.")
                    continue  # Skip this iteration
                
                success, buffer = cv2.imencode('.jpg', self.frame)
                if not success:
                    print("❌ Failed to encode frame.")
                    continue

                frame_bytes = buffer.tobytes()
                print("✅ Sending frame to browser.")

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    ### 📂 **SQLite Database Functions**
    def init_database(self):
        conn = sqlite3.connect(config["database_path"])
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp TEXT,
                          count INTEGER)''')
        conn.commit()
        conn.close()

    def log_detection(self, timestamp, count):
        conn = sqlite3.connect(config["database_path"])
        cursor = conn.cursor()
        cursor.execute("INSERT INTO detections (timestamp, count) VALUES (?, ?)", (timestamp, count))
        conn.commit()
        conn.close()
        print(f"Logged {count} humans at {timestamp}")

    ### 📧 **Email Alert Function**
    def send_alert(self, timestamp, human_count):
        if self.last_alert_time and (datetime.datetime.now() - self.last_alert_time).seconds < 60:
            return  # Prevent spam (1 email per minute)

        sender_email = config["sender_email"]
        sender_password = config["sender_password"]
        recipient_email = config["recipient_email"]

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = f"🚨 Alert: {human_count} Person(s) Detected at {timestamp}"

        body = f"A total of {human_count} human(s) were detected on {timestamp}."
        message.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
            server.quit()
            print(f"✅ Alert email sent to {recipient_email}")
            self.last_alert_time = datetime.datetime.now()  # Update last alert time
        except Exception as e:
            print(f"❌ Failed to send email: {e}")

# Initialize Human Detector
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
    app.run(host='0.0.0.0', port=5000)
