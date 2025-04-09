from django.urls import path
from .views import *

urlpatterns = [
    path('home/',MainPage.as_view(),name='main'),
    path('about/',About.as_view(),name='about'),
    path('registration/',RegView.as_view(),name='reg'),
    path('chatbot/',ChatbotView.as_view(),name='bot'),
    path('video_feed_object/', video_feed_object, name='video_feed_object'),
    path('object/', ObjectView.as_view(), name='object'),
    path('start-detection/', start_detection, name='start_detection'),
    path('stop-detection/', stop_detection, name='stop_detection'),
    path('forgot-password/',forgot_password, name='forgot_password'),
    path('password-reset-sent/',password_reset_sent, name='password_reset_sent'),
    path('reset-password/<uidb64>/<token>/',reset_password, name='reset_password'),
    path('password-reset-complete/',password_reset_complete, name='password_reset_complete'),
    path('profile/',profile_view, name='profile'),
    path('profile/update/',update_profile, name='update_profile'),
    path('profile/password/',change_password, name='change_password'),
]