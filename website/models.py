
from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), unique=True)
    filepath = db.Column(db.String(255))
    date_uploaded = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    processed_video = db.relationship('ProcessedVideo', backref='original_video', uselist=False)

class ProcessedVideo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), unique=True)
    filepath = db.Column(db.String(255))
    date_processed = db.Column(db.DateTime(timezone=True), default=func.now())
    original_video_id = db.Column(db.Integer, db.ForeignKey('video.id'))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    videos = db.relationship('Video', backref='user')