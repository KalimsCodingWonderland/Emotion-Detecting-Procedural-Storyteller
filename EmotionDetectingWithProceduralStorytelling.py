import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import pyttsx3
import threading
from collections import deque
import google.generativeai as genai
import time
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Initialize TTS Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate
engine.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)

# Configure Google Generative AI with your API key
api_key = "AIzaSyA3g28EZKiiYqXsNoMDUcjxBn4I85UwPes"  # Replace with your API key
genai.configure(api_key=api_key)

# Initialize the Generative Model
model = genai.GenerativeModel("gemini-1.5-flash")

# Global variables
locked_emotion = None
emotion_history = deque(maxlen=10)
emotion_locked = False
story_context = "The story begins..."  # Initial story context
lock_start_time = 0  # Track the start time of the emotion lock

# Function to preprocess the face: alignment and enhancement
def preprocess_face(frame, bbox):
    x, y, w, h = bbox
    face_img = frame[y:y + h, x:x + w]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.equalizeHist(face_img)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    return face_img

# Function to analyze the emotion of the detected face
def analyze_emotion(face_img):
    try:
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print("Emotion analysis failed:", e)
        return None

# Function to generate and narrate a story based on the detected emotion
def generate_and_narrate_story(emotion):
    global story_context
    # Generate a story prompt based on the current emotion and the ongoing story context
    emotion_prompt = {
        'happy': "Suddenly, a wave of joy and positivity changes the course of events dramatically.",
        'sad': "The atmosphere shifts into a deep melancholy, altering the path of the unfolding tale.",
        'angry': "Anger surges, and conflict erupts, steering the story into chaos and confrontation.",
        'surprise': "A shocking twist turns everything upside down, leaving everyone stunned.",
        'fear': "A sense of fear takes hold, darkening the mood and twisting the events towards dread.",
        'neutral': "The scene stabilizes for a moment, creating a calm yet tense anticipation.",
    }

    # Build the prompt with the ongoing context and the new emotional cue
    prompt = f"{story_context} {emotion_prompt.get(emotion, 'A sudden change shifts the narrative dramatically...')} Continue the story with this new turn."

    # Call the Google Generative AI model to generate a continuation of the story
    story = generate_story_from_google(prompt)
    story_context += " " + story  # Update the context to include the new segment

    # Narrate the story
    print(story)
    engine.say(story)
    engine.runAndWait()

# Function to generate a story using Google Generative AI
def generate_story_from_google(prompt):
    try:
        # Generate text using Google Generative AI model
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating story with Google Generative AI: {e}")
        return "There was an error generating the story."

# Function to reset the system to detect a new emotion
def reset_emotion():
    global emotion_locked, locked_emotion
    emotion_locked = False
    locked_emotion = None
    print("Emotion detection reset.")

# Main Video Capture and Emotion Detection
cap = cv2.VideoCapture(0)

# Run story generation in a separate thread
def story_thread():
    global lock_start_time
    while True:
        if locked_emotion:
            generate_and_narrate_story(locked_emotion)
            time.sleep(1)  # Short pause to allow continuous narrative updates

control_thread = threading.Thread(target=story_thread, daemon=True)
control_thread.start()

with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_img = preprocess_face(frame, (x, y, w, h))
                emotion = analyze_emotion(face_img)
                if emotion:
                    emotion_history.append(emotion)
                    stable_emotion = max(set(emotion_history), key=emotion_history.count)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, stable_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        if emotion_locked and locked_emotion:
            cv2.putText(frame, f"Locked Emotion: {locked_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Real-Time Emotion Storyteller', frame)

        # Enter key to lock emotion
        if cv2.waitKey(5) & 0xFF == 13:  # Enter key to lock emotion at any time
            if stable_emotion:
                locked_emotion = stable_emotion
                emotion_locked = True
                lock_start_time = time.time()  # Reset lock time
                print(f"Locked Emotion: {locked_emotion}")

        # 'u' key to manually unlock emotion
        if cv2.waitKey(5) & 0xFF == ord('u'):
            reset_emotion()

        # 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
