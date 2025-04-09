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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

app = Flask(__name__, static_folder='static')

# Load Configurations
with open('config.json') as config_file:
    config = json.load(config_file)

class HumanDetector:
    def __init__(self):
        # Configure model with performance settings
        model_config = {
            'model': config.get("model_weights_path", "yolov8n.pt"),
            'imgsz': (config.get('frame_height', 320), config.get('frame_width', 240)),
            'conf': config.get('detection_confidence', 0.5),
            'device': 'cpu'  # Use CPU for broader compatibility
        }
        
        self.model = YOLO(model_config['model'])
        self.model_config = model_config
        
        self.camera = None
        self.lock = threading.Lock()
        self.current_human_count = 0
        self.max_human_count = 0
        self.frame = None
        self.is_camera_running = False
        self.camera_index = None
        self.last_alert_time = None
        self.detection_active = False
        self.last_detection_count = 0

        # Initialize resources
        self.init_database()
        self.setup_error_frame()

    def setup_error_frame(self):
        """Create a default frame to display when camera fails"""
        self.error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.error_frame, 'Camera Initialization Failed', 
                    (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def init_database(self):
        """Initialize SQLite database for logging detections"""
        try:
            db_path = config["database_path"]
            db_dir = os.path.dirname(db_path)
            
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)

            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                              timestamp TEXT,
                              count INTEGER)''')
            conn.commit()
            conn.close()
            logging.info("Database initialized successfully")
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")

    def start_camera(self):
        """Parallel camera initialization with improved robustness"""
        if self.is_camera_running:
            return True

        if self.detection_active:
            return False

        logging.info("Attempting to initialize camera...")
        camera_indices = [0, 1, 2, 3, -1]
        
        with ThreadPoolExecutor(max_workers=len(camera_indices)) as executor:
            futures = {executor.submit(self.try_camera_index): index for index in camera_indices}
            
            for future in as_completed(futures):
                try:
                    camera, index = future.result()
                    if camera is not None:
                        logging.info(f"Camera initialized successfully at index {index}")
                        self.camera = camera
                        self.frame = camera.read()[1]
                        self.is_camera_running = True
                        self.camera_index = index
                        self.detection_active = True
                        
                        # Start detection thread
                        self.detection_thread = threading.Thread(
                            target=self.detect_loop, 
                            daemon=True
                        )
                        self.detection_thread.start()
                        return True
                except Exception as e:
                    logging.error(f"Camera initialization error: {e}")
        
        logging.error("Failed to initialize camera on all indices")
        self.camera = None
        self.frame = self.error_frame
        return False

    def try_camera_index(self):
        """Try to open a camera with optimized settings"""
        for index in [0, 1, 2, 3, -1]:
            try:
                camera = cv2.VideoCapture(index, cv2.CAP_ANY)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                camera.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for performance

                ret, frame = camera.read()
                if ret and frame is not None and not np.all(frame == 0):
                    return camera, index
                
                camera.release()
            except Exception as e:
                logging.warning(f"Camera index {index} error: {e}")
        
        return None, None

    def detect_loop(self):
        """Continuous detection loop with controlled frequency"""
        while self.detection_active and self.is_camera_running:
            try:
                if self.camera:
                    ret, frame = self.camera.read()
                    if ret:
                        self.frame = self.detect_humans(frame)
                    threading.Event().wait(0.2)  # 5 FPS equivalent
            except Exception as e:
                logging.error(f"Detection loop error: {e}")
                break

    def detect_humans(self, frame):
        """Optimized human detection method"""
        try:
            # Use model configuration from initialization
            results = self.model(frame, 
                                 imgsz=self.model_config['imgsz'], 
                                 conf=self.model_config['conf'])
            
            human_count = 0
            for result in results[0].boxes:
                class_id = int(result.cls)
                if class_id == 0:  # Person class
                    human_count += 1
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Thread-safe count updates
            with self.lock:
                self.current_human_count = human_count
                self.max_human_count = max(self.max_human_count, human_count)

            # Log and alert only on count change
            if human_count != self.last_detection_count:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.log_detection(timestamp, human_count)
                self.send_alert(timestamp, human_count)
                self.last_detection_count = human_count

            cv2.putText(frame, f'Humans: {human_count}', 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return frame
        except Exception as e:
            logging.error(f"Detection error: {e}")
            return frame

    def generate_frames(self):
        """Frame generator for video streaming"""
        while self.is_camera_running:
            with self.lock:
                if self.frame is None:
                    continue
                
                success, buffer = cv2.imencode('.jpg', self.frame)
                if not success:
                    continue

                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def log_detection(self, timestamp, count):
        """Log human detections to SQLite database"""
        try:
            conn = sqlite3.connect(config["database_path"], check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO detections (timestamp, count) VALUES (?, ?)", 
                           (timestamp, count))
            conn.commit()
            conn.close()
            logging.info(f"Logged {count} humans at {timestamp}")
        except Exception as e:
            logging.error(f"Detection logging error: {e}")

    def send_alert(self, timestamp, human_count):
        """Send email alert with rate limiting"""
        try:
            # Prevent alert spam
            if (self.last_alert_time and 
                (datetime.datetime.now() - self.last_alert_time).seconds < 60):
                return

            sender_email = config["sender_email"]
            sender_password = config["sender_password"]
            recipient_email = config["recipient_email"]

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_email
            message["Subject"] = f"🚨 Alert: {human_count} Person(s) Detected"

            body = f"A total of {human_count} human(s) were detected on {timestamp}."
            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)
                logging.info(f"Alert email sent to {recipient_email}")
                self.last_alert_time = datetime.datetime.now()

        except Exception as e:
            logging.error(f"Email alert failed: {e}")

    def stop_camera(self):
        """Safely stop camera and reset state"""
        with self.lock:
            self.detection_active = False
            if self.camera:
                self.camera.release()
                self.camera = None
            self.is_camera_running = False
            self.current_human_count = 0
            self.max_human_count = 0
            self.camera_index = None
            self.frame = self.error_frame
        return True

# Initialize Human Detector
detector = HumanDetector()

@app.route('/')
def index():
    """Main route to start camera and render template"""
    if not detector.start_camera():
        return "Camera initialization failed. Please check system configuration."
    return render_template('index.html')

@app.route('/stop_camera')
def stop_camera():
    """Route to stop camera"""
    detector.stop_camera()
    return jsonify({"status": "Camera stopped successfully"})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(detector.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get current detection statistics"""
    with detector.lock:
        return jsonify({
            'current_human_count': detector.current_human_count,
            'max_human_count': detector.max_human_count,
            'is_camera_running': detector.is_camera_running
        })

if __name__ == '__main__':
    logging.info("Starting Human Detection System")
    app.run(host='0.0.0.0', port=5000, debug=False)