from flask import Blueprint, render_template, request, flash, redirect, url_for, send_from_directory
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   ##means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
import cv2
import os
import mediapipe as mp
import numpy as np
import math
from object_detection import ObjectDetection


auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('views.lobby'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("login.html", user=current_user)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(
                password1, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Account created!', category='success')
            return redirect(url_for('views.lobby'))

    return render_template("sign_up.html", user=current_user)

# @auth.route("/", methods=["GET", "POST"])
# def predict_img():
#     if request.method == "POST":
#         if 'file' in request.files:
#             f = request.files['file']
#             basepath = os.path.dirname(__file__)
#             filepath = os.path.join(basepath, 'uploads', f.filename)
#             print("Upload folder is ", filepath)
#             f.save(filepath)
#             global imgpath
#             predict_img.imgpath = f.filename
#             print("Printing predict_img ::::::: ", predict_img.imgpath)
            
#             file_extension = f.filename.rsplit('.', 1)[1].lower()
            
#             if file_extension == 'mp4':
#                 video_path = filepath
#                 cap = cv2.VideoCapture(video_path)
                
#                 # get video dimensions
#                 frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
#                 # Add your video processing logic here
#                 # For example, performing object detection on each frame
#                 # yolo = YOLO('best.pt')
#                 # while cap.isOpened():
#                 #     ret, frame = cap.read()
#                 #     if not ret:
#                 #         break
#                 #     # Perform detection on the frame
#                 #     detections = yolo.predict(frame, save=True)
                
#                 cap.release()
#                 return display(f.filename)
#             else:
#                 return "Only mp4 video files are allowed", 400


# # The display function is used to serve the image or video from the folder_path directory.
# @auth.route('/<path:filename>')
# def display(filename):
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
#     directory = folder_path + '/' + latest_subfolder
#     print("Printing directory:", directory)
#     files = os.listdir(directory)
#     latest_file = files[0]

#     print(latest_file)

#     filename = os.path.join(folder_path, latest_subfolder, latest_file)
#     file_extension = filename.rsplit('.', 1)[1].lower()

#     environ = request.environ
#     if file_extension == 'jpg':
#         return send_from_directory(directory, latest_file, environ)  # Shows the result in separate tab
#     else:
#         return "Invalid file format"
