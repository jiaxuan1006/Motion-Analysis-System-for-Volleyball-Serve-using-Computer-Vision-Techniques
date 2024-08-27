from flask import Blueprint, render_template, request, flash, jsonify, current_app, redirect, url_for, send_file, send_from_directory, abort
from flask_login import login_required, current_user
from .models import Video, ProcessedVideo
from . import db
import os
import cv2
import mediapipe as mp
import numpy as np
import math
from object_detection import ObjectDetection

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


od = ObjectDetection()

views = Blueprint('views', __name__)

@views.context_processor
def inject_user():
    return dict(user=current_user)

@views.before_app_request
def before_request():
    global UPLOAD_FOLDER, DETECTED_FOLDER
    UPLOAD_FOLDER = os.path.join(current_app.root_path, 'static', 'uploads')
    DETECTED_FOLDER = os.path.join(current_app.root_path, 'static', 'results')

@views.route('/lobby')
@login_required
def lobby():
    user = current_user if current_user.is_authenticated else None
    return render_template("lobby.html", user=user)

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            if f.filename == '':
                flash('No file selected!', category='error')
            elif not f.filename.lower().endswith('.mp4'):
                flash('Only MP4 files are allowed!', category='error')
            else:
                filename = f.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                f.save(filepath)

                # Normalize the file path
                relative_filepath = os.path.join('uploads', filename).replace('\\', '/')
                new_video = Video(filename=filename, filepath=relative_filepath, user_id=current_user.id)
                db.session.add(new_video)
                db.session.commit()
                flash('Video uploaded successfully!', category='success')
        else:
            flash('No file uploaded!', category='error')

    user_videos = Video.query.filter_by(user_id=current_user.id).all()
    return render_template("home.html", user=current_user, videos=user_videos)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_velocity(p1, p2, time_elapsed):
    """Calculate the velocity between two points."""
    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return distance / time_elapsed

def get_phase(shoulder_elbow_wrist_angle, elbow_position, shoulder_position, wrist_position, head_position, wrist_velocity, current_state):
    if current_state == "preparation" and elbow_position[0] < shoulder_position[0] and 85 <= shoulder_elbow_wrist_angle <= 140:
        return "backswing"
    elif current_state == "backswing" and wrist_velocity > 0 and 150 <= shoulder_elbow_wrist_angle <= 180:
        return "forward_swing"
    elif current_state == "forward_swing" and wrist_velocity < 0 and 170 <= shoulder_elbow_wrist_angle <= 180 and wrist_position[1] > head_position[1]:
        return "follow_through"
    else:
        return current_state  # No change in state

@views.route('/process-video/<int:video_id>', methods=['GET'])
@login_required
def process_video(video_id):
    video = Video.query.get(video_id)
    if video and video.user_id == current_user.id:
        filepath = os.path.join(current_app.root_path, 'static', video.filepath)

        cap = cv2.VideoCapture(filepath)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        processed_filename = f'processed_{video.filename}'
        processed_filepath = os.path.join(current_app.root_path, 'static', 'results', processed_filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(processed_filepath, fourcc, 30.0, (frame_width, frame_height))

        # Initialize tracking variables
        tracking_objects = {}
        track_id = 0
        center_points_prev_frame = []

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            current_state = "preparation"
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                center_points_cur_frame = []

                # Object detection
                (class_ids, scores, boxes) = od.detect(frame)
                for box in boxes:
                    (x, y, w, h) = box
                    cx = int((x + x + w) / 2)
                    cy = int((y + y + h) / 2)
                    center_points_cur_frame.append((cx, cy))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Object tracking
                if len(center_points_prev_frame) == 0:
                    for pt in center_points_cur_frame:
                        tracking_objects[track_id] = pt
                        track_id += 1
                else:
                    tracking_objects_copy = tracking_objects.copy()
                    center_points_cur_frame_copy = center_points_cur_frame.copy()

                    for object_id, pt2 in tracking_objects_copy.items():
                        object_exists = False
                        for pt in center_points_cur_frame_copy:
                            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                            if distance < 20:
                                tracking_objects[object_id] = pt
                                object_exists = True
                                if pt in center_points_cur_frame:
                                    center_points_cur_frame.remove(pt)
                                continue
                        if not object_exists:
                            tracking_objects.pop(object_id)

                    for pt in center_points_cur_frame:
                        tracking_objects[track_id] = pt
                        track_id += 1


                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    head = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                            landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                    shoulder_elbow_wrist_angle = calculate_angle(shoulder, elbow, wrist)

                    # Calculate wrist velocity (simple difference, more sophisticated methods can be used)
                    if 'prev_wrist' in locals():
                        wrist_velocity = wrist[0] - prev_wrist[0]
                    else:
                        wrist_velocity = 0
                    prev_wrist = wrist

                    current_state = get_phase(shoulder_elbow_wrist_angle, elbow, shoulder, wrist, head, wrist_velocity, current_state)

                    # Define phase angles for evaluation
                    PHASES = {
                        "preparation": {"SEW": (160, 180), "HKA": (160, 180)},
                        "backswing": {"SEW": (85, 140), "HKA": (140, 170)},
                        "forward_swing": {"SEW": (150, 180), "HKA": (160, 180)},
                        "follow_through": {"SEW": (170, 180), "HKA": (160, 180)}
                    }

                    phase_angles = PHASES[current_state]

                     # Initialize feedback variables
                    sew_feedback = ""

                 # Feedback based on SEW angle
                    if phase_angles["SEW"][0] <= shoulder_elbow_wrist_angle <= phase_angles["SEW"][1]:
                        sew_color = (0, 255, 0)  # Green for good
                        sew_feedback = "Good"
                    else:
                        sew_color = (0, 0, 255)  # Red for poor
                        if current_state == "preparation":
                            if shoulder_elbow_wrist_angle < phase_angles["SEW"][0]:
                                sew_feedback = "Poor. Try to extend your shoulder further during preparation."
                            elif shoulder_elbow_wrist_angle > phase_angles["SEW"][1]:
                                sew_feedback = "Poor. Reduce shoulder extension during preparation."
                        elif current_state == "backswing":
                            if shoulder_elbow_wrist_angle < phase_angles["SEW"][0]:
                                sew_feedback = "Poor. Increase your backswing angle."
                            elif shoulder_elbow_wrist_angle > phase_angles["SEW"][1]:
                                sew_feedback = "Poor. Decrease your backswing angle."
                        elif current_state == "forward_swing":
                            if shoulder_elbow_wrist_angle < phase_angles["SEW"][0]:
                                sew_feedback = "Poor. Try to extend your shoulder extension further for rotation momentum."
                            elif shoulder_elbow_wrist_angle > phase_angles["SEW"][1]:
                                sew_feedback = "Poor. Decrease your forward swing angle."
                        elif current_state == "follow_through":
                            if shoulder_elbow_wrist_angle < phase_angles["SEW"][0]:
                                sew_feedback = "Poor. Extend your arm further in the follow-through."
                            elif shoulder_elbow_wrist_angle > phase_angles["SEW"][1]:
                                sew_feedback = "Poor. Control your arm extension in the follow-through."


                    cv2.putText(frame, f"Phase: {current_state}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Right Shoulder-Elbow-Wrist (SEW) Angle: {shoulder_elbow_wrist_angle:.2f} - {sew_feedback}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    cv2.putText(frame, str(shoulder_elbow_wrist_angle),
                        tuple(np.multiply(elbow, [500, 700]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
                    # Add "right shoulder angle" text
                 
                    
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                out.write(frame)

        cap.release()
        out.release()

        new_processed_video = ProcessedVideo(filename=processed_filename, filepath=f'results/{processed_filename}', original_video_id=video.id)
        db.session.add(new_processed_video)
        db.session.commit()

        return redirect(url_for('views.result', filename=processed_filename))

    flash('Video not found or not authorized!', category='error')
    return redirect(url_for('views.home'))

@views.route('/result/<filename>')
@login_required
def result(filename):
    return render_template('result.html', filename=filename)

@views.route('/delete-video', methods=['POST'])
@login_required
def delete_video():
    data = request.json
    video_id = data.get('videoId')
    if video_id:
        video = Video.query.get(video_id)
        if video and video.user_id == current_user.id:
            db.session.delete(video)
            db.session.commit()
            return jsonify({'message': 'Video deleted successfully'})
    return jsonify({'error': 'Failed to delete video'})

@views.route('/analyzed-results')
@login_required
def analyzed_results():
    processed_videos = ProcessedVideo.query.join(Video, ProcessedVideo.original_video_id == Video.id)\
                                           .filter(Video.user_id == current_user.id).all()
    return render_template('analyzed_result.html', processed_videos=processed_videos)


@views.route('/download-result/<filename>')
@login_required
def download_result(filename):
    processed_video = ProcessedVideo.query.filter_by(filename=filename).first()
    if processed_video:
        file_path = os.path.join(current_app.root_path, 'static', processed_video.filepath)
        return send_file(file_path, as_attachment=True)
    flash('Processed video not found!', category='error')
    return redirect(url_for('views.analyzed_results'))

@views.route('/delete-result', methods=['POST'])
@login_required
def delete_result():
    data = request.json
    filename = data.get('filename')
    if filename:
        processed_video = ProcessedVideo.query.filter_by(filename=filename).first()
        if processed_video:
            db.session.delete(processed_video)
            db.session.commit()
            return jsonify({'message': 'Processed video deleted successfully'})
    return jsonify({'error': 'Failed to delete processed video'})


# @views.route('/compare-video/<filename>', methods=['GET'])
# @login_required
# def compare_video(filename):
#     original_video = ProcessedVideo.query.filter_by(filename=filename).first()

#     if not original_video or original_video.original_video.user_id != current_user.id:
#         flash('Video not found or not authorized!', category='error')
#         return redirect(url_for('views.home'))

#     other_videos = ProcessedVideo.query.filter(ProcessedVideo.filename != filename, ProcessedVideo.original_video.has(user_id=current_user.id)).all()

#     return render_template('compare_video.html', original_filename=filename, other_videos=other_videos)


@views.route('/results/<path:filename>')
def download_file(filename):
    file_path = os.path.join(current_app.root_path, 'static', 'results', filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        abort(404)
    return send_from_directory('static/results', filename)

@views.route('/show-frames/<filename>', methods=['GET'])
@login_required
def show_frames(filename):
    video = ProcessedVideo.query.filter_by(filename=filename).first()
    if video:
        filepath = os.path.join(current_app.root_path, 'static', 'results', filename)
        
        # Extract frames from the video
        cap = cv2.VideoCapture(filepath)
        frame_count = 0
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Save each frame as a temporary file and store its path in the frames list
            frame_filename = f'static/results/frames/{filename}_frame_{frame_count}.jpg'
            save_path = os.path.join(current_app.root_path, frame_filename)
            cv2.imwrite(save_path, frame)
            frames.append(f'results/frames/{filename}_frame_{frame_count}.jpg')
            frame_count += 1
        
        cap.release()

        return render_template('frames.html', frames=frames, filename=filename)
    
    flash('Video not found!', category='error')
    return redirect(url_for('views.home'))