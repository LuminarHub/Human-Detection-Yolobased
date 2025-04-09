from django import forms
from .models import *

from django.contrib.auth.forms import UserCreationForm


class LogForm(forms.Form):
    email=forms.EmailField(widget=forms.TextInput(attrs={"placeholder":"Username","class":"form-control","style":"border-radius: 0.75rem; "}))
    password=forms.CharField(widget=forms.PasswordInput(attrs={"placeholder":"Password","class":"form-control","style":"border-radius: 0.75rem; "}))


class Reg(UserCreationForm):
    class Meta:
        model=CustUser
        fields=['email','username','password1','password2']   


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = CustUser
        fields = ['username', 'email']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
        }

from django.contrib.auth.forms import PasswordResetForm as DjangoPasswordResetForm
from django.contrib.auth.forms import SetPasswordForm as DjangoSetPasswordForm

class PasswordResetForm(forms.Form):
    email = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter your email address'})
    )

class SetPasswordForm(DjangoSetPasswordForm):
    new_password1 = forms.CharField(
        label="New password",
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Enter new password'})
    )
    new_password2 = forms.CharField(
        label="Confirm new password",
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm new password'})
    )
