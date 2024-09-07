# Real-Time Emotion-Based Storyteller

An AI-powered interactive application that detects user emotions using facial recognition and generates dynamic, real-time narratives. This project combines OpenCV, MediaPipe, DeepFace, Google Generative AI, and pyttsx3 to create an engaging and adaptive storytelling experience.

## Features

- **Real-Time Emotion Detection**: Uses OpenCV and MediaPipe for facial recognition and DeepFace for analyzing user emotions.
- **Dynamic Story Generation**: Google Generative AI generates unique story continuations based on the detected emotions.
- **Interactive Narration**: Uses pyttsx3 to narrate the story aloud in real-time.
- **Emotion Locking Mechanism**: Allows emotions to lock into the storyline, guiding the narrative based on consistent user expressions.

## Usage Instructions

- **Lock Emotions**: Press Enter to lock the current emotion into the story, influencing the narrative direction.
- **Unlock Emotions**: Press ‘u’ to unlock and reset, allowing new emotions to be detected.
- **Queue Emotions**: You can lock a new emotion while waiting for the current story segment to finish, enabling continuous narrative updates.
- **Quit**: Press ‘q’ to exit/quit the application.


## Technologies Used

- **Python**: Core programming language.
- **OpenCV**: For video capture and real-time face detection.
- **MediaPipe**: For enhanced facial recognition and landmark detection.
- **DeepFace**: For emotion analysis from facial expressions.
- **Google Generative AI**: Generates narrative content based on emotions.
- **pyttsx3**: Text-to-speech engine for narration.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Real-Time-Emotion-Storyteller.git
   cd Real-Time-Emotion-Storyteller
