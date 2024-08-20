import cv2
from deepface import DeepFace
import pygame
import mingus.core.notes as notes
import mingus.midi.midi_file_out as midi_file_out
from mingus.containers import Bar, Track

# Initialize Pygame mixer
pygame.mixer.init()

# Define musical patterns based on emotions
emotion_music_patterns = {
    'happy': ['C', 'E', 'G', 'C5'],
    'sad': ['A', 'C', 'E', 'A4'],
    'angry': ['D', 'F', 'A', 'D5'],
    'surprise': ['G', 'B', 'D', 'G5'],
    'neutral': ['C', 'D', 'E', 'G'],
}


def generate_music(emotion):
    """
    Generate a simple melody based on the detected emotion.
    """
    pattern = emotion_music_patterns.get(emotion, emotion_music_patterns['neutral'])
    track = Track()
    bar = Bar()

    for note in pattern:
        if notes.is_valid_note(note):
            bar + note
        else:
            print(f"Invalid note found: {note}")

    track + bar

    # Save the track as a MIDI file
    midi_filename = f'{emotion}_music.mid'
    midi_file_out.write_Track(midi_filename, track)
    return midi_filename


def play_music(midi_filename):
    """
    Play the generated MIDI music using Pygame.
    """
    try:
        pygame.mixer.music.load(midi_filename)
        pygame.mixer.music.play()
        print(f"Playing {midi_filename}...")
    except pygame.error as e:
        print(f"Error playing music: {e}")


# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Minimum confidence threshold for detecting emotions
CONFIDENCE_THRESHOLD = 0.6

# Create a resizable window and set a default size
cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Emotion Detection', 1280, 720)  # Initial window size

current_emotion = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Get the size of the window (in pixels)
    window_size = cv2.getWindowImageRect('Emotion Detection')
    window_width = window_size[2]  # width
    window_height = window_size[3]  # height

    # Calculate the scaling factors
    (frame_height, frame_width) = frame.shape[:2]
    x_scale = window_width / frame_width
    y_scale = window_height / frame_height

    # Resize the frame to fit the window size
    frame_resized = cv2.resize(frame, (window_width, window_height))

    try:
        # Use DeepFace to analyze the frame for emotion
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion]

            if confidence >= CONFIDENCE_THRESHOLD and emotion != current_emotion:
                face_info = result['region']
                x, y, w, h = face_info['x'], face_info['y'], face_info['w'], face_info['h']

                # Adjust face bounding box coordinates to match resized frame
                x = int(x * x_scale)
                y = int(y * y_scale)
                w = int(w * x_scale)
                h = int(h * y_scale)

                # Draw a rectangle around the face
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the emotion label and confidence on the frame
                label = f'{emotion.capitalize()} ({confidence * 100:.2f}%)'
                cv2.putText(frame_resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Generate and play music based on the detected emotion
                midi_filename = generate_music(emotion)
                play_music(midi_filename)

                current_emotion = emotion  # Update the current emotion to avoid repetitive playback

            else:
                # Display "Neutral" if confidence is too low
                cv2.putText(frame_resized, 'Neutral', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)

        else:
            cv2.putText(frame_resized, 'No Face Detected', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

    except Exception as e:
        print(f"Error detecting emotion: {e}")
        cv2.putText(frame_resized, 'Error', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the resized frame in the window
    cv2.imshow('Emotion Detection', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
