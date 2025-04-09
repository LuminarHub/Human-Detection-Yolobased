from django.shortcuts import render, redirect
from django.views.generic import TemplateView, FormView, CreateView, View
from django.urls import reverse_lazy
from django.contrib.auth import authenticate, login, logout
from .models import *
from .forms import *
from django.http import HttpResponseBadRequest
from django.http import JsonResponse, HttpResponseNotAllowed
from groq import Groq
from django.views.decorators.csrf import csrf_exempt
import json
import re
import cv2
import numpy as np

from django.http import StreamingHttpResponse, HttpResponse
from ultralytics import YOLO
import threading
import time


class LoginView(FormView):
    template_name = "login.html"
    form_class = LogForm

    def post(self, request, *args, **kwargs):
        log_form = LogForm(data=request.POST)
        if log_form.is_valid():
            us = log_form.cleaned_data.get('email')
            ps = log_form.cleaned_data.get('password')
            request.session['email'] = us
            user = authenticate(request, email=us, password=ps)
            if user:
                login(request, user)
                if user.is_superuser == 1 :
                   return redirect('admin:index')
                else:
                     return redirect('main')
            else:
                return render(request, 'login.html', {"form": log_form})
        else:
            return render(request, 'login.html', {"form": log_form})

class RegView(CreateView):
    form_class = Reg
    template_name = "register.html"
    model = CustUser
    success_url = reverse_lazy("log")


class MainPage(TemplateView):
    template_name = 'main.html'

class About(TemplateView):
    template_name = 'about.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['data'] = About.objects.all().first()
        return context 


from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

@method_decorator(csrf_exempt, name='dispatch')
class CustomLogoutView(View):
    def get(self, request):
        logout(request)  # Log out the user
        # del request.session['email']
        response = redirect('log')  

        return response


def custom_logout(request):
    logout(request)
    return redirect('log')






import cv2
import threading
import time
import numpy as np
from django.http import StreamingHttpResponse
from django.views.generic import TemplateView
from ultralytics import YOLO
from django.core.mail import send_mail
from django.conf import settings
from datetime import datetime
# Load the YOLO model globally
model = YOLO("yolov8n.pt")

# Lock for thread safety
lock = threading.Lock()

# email_time_period = 60


import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from django.core.mail import EmailMultiAlternatives
from django.conf import settings
from ultralytics import YOLO

class HumanDetector:
    def __init__(self, model_path='yolov8n.pt', email_time_period=10):
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise ValueError("Failed to load YOLO model")

        self.video = None
        self.is_running = False
        self.current_frame = None
        self.detection_active = False

        self.lock = threading.Lock()
        self.unique_humans_count = 0

        # Email-related properties
        self.last_email_sent_time = None
        self.email_time_period = email_time_period

        # Tracking human detection state
        self.humans_in_frame = 0
        self.last_human_count = 0
        
        # Store detection frames and timestamps
        self.detection_frames = []  # List to store all detection screenshots
        self.detection_timestamps = []
        
        # Simple human tracking
        self.last_detection_time = None
        self.detection_cooldown = 3  # seconds between detection events 
    
    def start_camera(self):
        """Initialize camera."""
        if self.is_running:
            return True

        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Failed to open camera")
            return False

        self.is_running = True

        # Start frame capture thread
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

        return True

    def _capture_loop(self):
        """Continuously capture frames."""
        while self.is_running:
            if self.video is None or not self.video.isOpened():
                time.sleep(1)
                continue

            success, frame = self.video.read()
            if success:
                with self.lock:
                    self.current_frame = frame

                if self.detection_active:
                    self.detect_humans(frame)
            else:
                print("Failed to read frame")

            time.sleep(0.1)

    def detect_humans(self, frame):
        """Detect humans in frame and count each individual."""
        try:
            results = self.model(frame)
            human_count = 0
            human_boxes = []

            # Get all human detections with their bounding boxes
            for result in results[0].boxes:
                class_id = int(result.cls)
                confidence = result.conf.item()
                if class_id == 0 and confidence > 0.5:  # Person class with confidence > 0.5
                    human_count += 1
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    human_boxes.append((x1, y1, x2, y2, confidence))
            
            current_time = datetime.now()
            
            # Check if we have new humans or significant time has passed since last detection
            if human_count > 0 and (
                human_count > self.humans_in_frame or 
                self.last_detection_time is None or 
                (current_time - self.last_detection_time).total_seconds() > self.detection_cooldown
            ):
                # Create a copy of the frame for annotation
                annotated_frame = frame.copy()
                
                # Draw bounding boxes around each human
                for i, (x1, y1, x2, y2, conf) in enumerate(human_boxes):
                    # Different color for each human in the frame
                    color = (0, 0, 255)  # Red by default
                    if i % 3 == 1:
                        color = (0, 255, 0)  # Green for second human
                    elif i % 3 == 2:
                        color = (255, 0, 0)  # Blue for third human
                        
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f'Human #{i+1} ({conf:.2f})',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add total humans detected in this frame
                timestamp = datetime.now()
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                # Update humans count and save the image
                new_humans = max(0, human_count - self.humans_in_frame)
                if new_humans > 0 or self.humans_in_frame == 0:
                    # Increase the total count by the number of new humans
                    self.unique_humans_count += human_count
                    
                    # Add text with detection info to the image
                    cv2.putText(annotated_frame, f"Detection - {timestamp_str}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Humans in frame: {human_count}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Total humans detected: {self.unique_humans_count}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Store the annotated frame
                    self.detection_frames.append(annotated_frame)
                    self.detection_timestamps.append(timestamp_str)
                    
                    print(f"Detection at {timestamp_str}: {human_count} humans in frame, total detected: {self.unique_humans_count}")
                    
                    # Update last detection time
                    self.last_detection_time = current_time
                
                # Update humans in frame count
                self.humans_in_frame = human_count
                self.last_human_count = human_count
            
            # Reset count when no humans in frame
            if human_count == 0:
                self.humans_in_frame = 0

            return True
        except Exception as e:
            print(f"Error in human detection: {e}")
            return False

    def get_frame(self):
        """Process and return the frame with YOLO detections."""
        with self.lock:
            if self.current_frame is None:
                # Return a blank frame if no frame is available
                blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                _, jpeg = cv2.imencode('.jpg', blank)
                return jpeg.tobytes()

            frame_to_process = self.current_frame.copy()

            if self.detection_active:
                try:
                    results = self.model(frame_to_process)
                    human_count = 0
                    
                    for i, result in enumerate(results[0].boxes):
                        class_id = int(result.cls)
                        confidence = result.conf.item()
                        if class_id == 0 and confidence > 0.6:  # Person class only, confidence > 0.6
                            human_count += 1
                            x1, y1, x2, y2 = map(int, result.xyxy[0])
                            
                            # Different color for each human
                            color = (0, 0, 255)  # Red by default
                            if i % 3 == 1:
                                color = (0, 255, 0)  # Green for second human
                            elif i % 3 == 2:
                                color = (255, 0, 0)  # Blue for third human
                                
                            cv2.rectangle(frame_to_process, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame_to_process, f'Human #{i+1} ({confidence:.2f})',
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    annotated_frame = frame_to_process
                    
                except Exception as e:
                    print(f"Error in YOLO detection: {e}")
                    annotated_frame = frame_to_process
                    cv2.putText(annotated_frame, f"Error: {str(e)}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                annotated_frame = frame_to_process

            # Add detection status to the frame
            cv2.putText(annotated_frame, f"Total humans detected: {self.unique_humans_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', annotated_frame)
            return jpeg.tobytes()

    def generate_frames(self):
        """Generate frames for video streaming."""
        if not self.is_running:
            self.start_camera()

        while self.is_running:
            frame = self.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)

    def start_detection(self):
        """Start the human detection process."""
        if self.detection_active:
            print("Detection is already active")
            return False

        if not self.is_running:
            if not self.start_camera():
                print("Failed to start camera")
                return False

        # Reset counters and lists
        self.unique_humans_count = 0
        self.humans_in_frame = 0
        self.last_human_count = 0
        self.detection_frames = []
        self.detection_timestamps = []
        self.last_detection_time = None

        print("Starting human detection")
        self.detection_active = True

        return True

    def stop_detection(self, email=None):
        """Stop the detection process and send email with all screenshots if applicable."""
        if not self.detection_active:
            print("Detection is not active")
            return False

        self.detection_active = False

        print(f"Human detection stopped. Total unique humans detected: {self.unique_humans_count}")
        
        # Send email with all screenshots if email is provided and we have detections
        if email and self.unique_humans_count > 0:
            self._send_email_with_screenshots(email)

        return True
    
    def _send_email_with_screenshots(self, email):
        """Send email with all detection screenshots."""
        try:
            subject = f"Human Detection Alert - {self.unique_humans_count} Humans Detected"
            text_content = f"Total humans detected: {self.unique_humans_count}\n\n"
            
            # Add detection timestamps
            if self.detection_timestamps:
                text_content += "Detection timestamps:\n"
                for i, timestamp in enumerate(self.detection_timestamps, 1):
                    text_content += f"{i}. {timestamp}\n"
            
            # Create message container
            msg = EmailMultiAlternatives(
                subject=subject,
                body=text_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[email]
            )
            
            # Add HTML content
            html_content = f"""
            <html>
            <body>
                <h1>Human Detection Alert</h1>
                <p><strong>Total humans detected:</strong> {self.unique_humans_count}</p>
                
                <h2>Detection events ({len(self.detection_timestamps)}):</h2>
                <ul>
                    {"".join(f"<li>Detection {i+1}: {timestamp}</li>" for i, timestamp in enumerate(self.detection_timestamps))}
                </ul>
                
                <p>Please see the attached screenshots of each detection event.</p>
            </body>
            </html>
            """
            msg.attach_alternative(html_content, "text/html")
            
            # Attach all screenshots
            for i, frame in enumerate(self.detection_frames):
                # Convert the image to JPEG
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()
                
                # Create a unique filename for each detection
                clean_timestamp = self.detection_timestamps[i].replace(" ", "_").replace(":", "-")
                filename = f'detection_{i+1}_{clean_timestamp}.jpg'
                
                # Attach the image
                img_attachment = MIMEImage(img_bytes)
                img_attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                msg.attach(img_attachment)
            
            # Send the email
            msg.send()
            print(f"Email with {len(self.detection_frames)} screenshots sent to {email}")
            return True
            
        except Exception as e:
            print(f"Error sending email with screenshots: {e}")
            return False

    def stop_camera(self):
        """Stop the camera and cleanup."""
        self.is_running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.video:
            self.video.release()

detector = HumanDetector()

def video_feed_object(request):
    """Django view for video streaming."""
    return StreamingHttpResponse(
        detector.generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

class ObjectView(TemplateView):
    """View for rendering the object detection page."""
    template_name = 'object.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['data'] = Userdetails.objects.all().first()
        return context  

def start_detection(request):
    """Start the detection process."""
    success = detector.start_detection()
    message = "Detection started" if success else "Failed to start detection"
    return JsonResponse({'status': message})

# def stop_detection(request):
#     """Stop the detection and send email with all screenshots."""
#     user_email = request.user.email
#     detection_count = detector.unique_humans_count
#     screenshot_count = len(detector.detection_frames)
    
#     success = detector.stop_detection(email=user_email)
#     detector.stop_camera()
#     message = f"Detection stopped. {detection_count} humans detected with {screenshot_count} screenshots sent to {user_email}"
#     if not success:
#         message = "Failed to stop detection or send email"
        
#     return JsonResponse({'status': message})

def stop_detection(request):
    """Stop the detection and stop the video feed."""
    user_email = request.user.email
    detector.stop_detection(email=user_email)
    # Don't stop the camera immediately to allow viewing the last frame
    detector.stop_camera()  # Commented out to keep the video feed running
    return JsonResponse({'status': 'Detection stopped. Email sent with detection results.'})



class ChatbotView(View):
    def get(self, request):
        return render(request, "chatbot.html")
    def post(self, request): 
        try:
            body = json.loads(request.body)
            user_input = body.get('userInput')
        except json.JSONDecodeError as e:
            return JsonResponse({"error": "Invalid JSON format."})
    
        if not user_input:  # If user_input is None or empty
            print("no")
            return JsonResponse({"error": "No user input provided."})  
        
        print("User Input:", user_input)
        
        static_responses = {
            # "hi": "Hello! How can I assist you today?",
            # "hello": "Hi there! How can I help you?",
            # "how are you": "I'm just a chatbot, but I'm doing great! How about you?",
            # "bye": "Goodbye! Take care.",
            # "whats up": "Not much, just here to help you with  queries. How can I help you today?",
        }

        lower_input = user_input.lower().strip()
        if lower_input in static_responses:
            print(static_responses[lower_input])
            return JsonResponse({'response': static_responses[lower_input]})
        
        try:
            print("Processing via GROQ")
            data = get_groq_response(user_input)
            print(data)
            treatment_list = data.split('\n')
            return JsonResponse({'response': treatment_list})
        except Exception as e:
            return JsonResponse({"error": f"Failed to get GROQ response: {str(e)}"})

client = Groq(api_key="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")



def get_groq_response(user_input):
    """
    Communicate with the GROQ chatbot to get a response based on user input.
    
    """
    system_prompt = {
      "role": "system",
      "content":
      "You are a helpful assistant"
    }

    chat_history = [system_prompt]
    while True:
        print("groq",user_input)
        chat_history.append({"role": "user", "content": user_input})

        chat_completion = client.chat.completions.create(model="llama3-70b-8192",
                                                messages=chat_history,
                                                max_tokens=100,
                                                temperature=1.2)

        chat_history.append({
        "role": "assistant",
        "content": chat_completion.choices[0].message.content
        })
        print("response",chat_completion.choices[0].message.content)
        response = chat_completion.choices[0].message.content
        response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)   
        # response = response.replace('+', ' and ').replace('.', ' there')
        return response

    

from django.core.mail import send_mail, BadHeaderError
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth import get_user_model
from .forms import PasswordResetForm, SetPasswordForm
from django.contrib import messages

def forgot_password(request):
    if request.method == 'POST':
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            try:
                user = CustUser.objects.get(email=email)
                subject = "Password Reset Request"
                
                # Create the reset token and encoded user ID
                token = default_token_generator.make_token(user)
                uid = urlsafe_base64_encode(force_bytes(user.pk))
                
                # Build the reset URL
                domain = request.get_host()
                protocol = 'https' if request.is_secure() else 'http'
                reset_url = f"{protocol}://{domain}/reset-password/{uid}/{token}/"
                
                # Create email content
                email_template = "password_reset_email.html"
                email_context = {
                    'user': user,
                    'reset_url': reset_url,
                    'domain': domain,
                }
                email_content = render_to_string(email_template, email_context)
                
                # Send email
                try:
                    send_mail(
                        subject,
                        "Please see the HTML content for password reset instructions.",
                        'noreply@humandetector.com',  # sender email
                        [email],
                        html_message=email_content,
                        fail_silently=False
                    )
                    return redirect('password_reset_sent')
                except BadHeaderError:
                    return HttpResponse('Invalid header found.')
                
            except CustUser.DoesNotExist:
                # We still redirect to the same success page even if the email doesn't exist
                # This prevents email enumeration attacks
                messages.error(request, "If an account exists with that email, a password reset link will be sent.")
                return redirect('password_reset_sent')
    else:
        form = PasswordResetForm()
    return render(request, 'forgot_password.html', {'form': form})

def password_reset_sent(request):
    return render(request, 'password_reset_sent.html')

def reset_password(request, uidb64, token):
    User = get_user_model()
    try:
        # Decode the uid to get the user's primary key
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
        
        # Verify the token is valid
        if default_token_generator.check_token(user, token):
            if request.method == 'POST':
                form = SetPasswordForm(user, request.POST)
                if form.is_valid():
                    form.save()
                    messages.success(request, "Your password has been reset successfully!")
                    return redirect('password_reset_complete')
            else:
                form = SetPasswordForm(user)
            return render(request, 'reset_password.html', {'form': form})
        else:
            messages.error(request, "The reset link is invalid or has expired.")
            return redirect('forgot_password')
            
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        messages.error(request, "The reset link is invalid or has expired.")
        return redirect('forgot_password')

def password_reset_complete(request):
    return render(request, 'password_reset_complete.html')


from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm

def profile_view(request):
    """View user profile information"""
    user = request.user
    return render(request, 'profile.html', {'user': user})

def update_profile(request):
    """Update user profile information"""
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated successfully!')
            return redirect('profile')
    else:
        form = UserProfileForm(instance=request.user)
    
    return render(request, 'update_profile.html', {'form': form})

def change_password(request):
    """Change user password"""
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            # Keep the user logged in after password change
            update_session_auth_hash(request, user)
            messages.success(request, 'Your password was successfully updated!')
            return redirect('profile')
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        form = PasswordChangeForm(request.user)
    
    return render(request, 'change_password.html', {'form': form})