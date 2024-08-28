# Motion-Analysis-System-for-Volleyball-Serve-using-Computer-Vision-Techniques

This is a project to develop a motion analysis system for volleyball overhand serve. 
The system is developed using computer vision techniques such as object detection tracking using YOLOv4 , human pose estimation using MediaPipe Pose model to calculate the joint angle between right shoulder-elbow=wrist if the volleyball player, detect specific phases in an overhand serve motion and visualize the result feedback based on the motion detected. 
The project is developed using OpenCV Python.
The website of the system is developed using Flask Python, DATABASE stored using Flask-sqlalchemy.


## Installation and Setup

```bash
Installed Python version 3.11 or above.
```

```bash
git clone <repo-url>
```

```bash
pip install -r requirements.txt
```

## Running The App

```bash
python main.py
```

## Viewing The App

Go to `http://127.0.0.1:5000`

## Process volleyball serve motion analysis

Select one of the videos in Examples and upload to the system. 

## Example of the Result of Analysis

![image](https://github.com/user-attachments/assets/4982575a-40a4-48bf-ae93-0ae444b4033c)
![image](https://github.com/user-attachments/assets/b14e93f5-be94-493f-8861-cdc37ede7c02)
![image](https://github.com/user-attachments/assets/b235d917-dc16-4ed4-89fb-249381908cb7)
![image](https://github.com/user-attachments/assets/df2f6d01-45b9-4216-9526-a96da0057dd7)



