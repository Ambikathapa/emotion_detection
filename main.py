import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tensorflow import keras
from PIL import Image, ImageTk

# Load the trained model
model = keras.models.load_model("facial_emotion_model.h5")

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create a VideoCapture object to capture video from the webcam
cap = cv2.VideoCapture(0)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Flags to control video processing
process_video = False
pause_video = False  # Flag to pause processing temporarily

# Function to start processing the video
def start_video_processing():
    global process_video, pause_video
    if not process_video:
        process_video = True  # Set the global flag
        pause_video = False  # Unpause processing
        process_emotion_video()
        # Hide the large font label
        title_label_large.pack_forget()
        # Show the normal font label
        title_label_normal.pack(pady=20)
        # Hide the "Start" and "Quit" buttons from the first frame
        start_button.pack_forget()
        quit_button.pack_forget()
        # Show the "Resume," "Pause," and "Quit" buttons on the second frame
        resume_button.pack(side="left", padx=20)
        pause_button.pack(side="left", padx=20)
        quit_button_second.pack(side="left", padx=20)

# Function to stop processing the video
def stop_video_processing():
    global process_video, pause_video
    process_video = False
    pause_video = False  # Ensure processing is not paused

# Function to pause processing temporarily
def pause_video_processing():
    global pause_video
    pause_video = True

# Function to resume processing
def resume_video_processing():
    global pause_video
    pause_video = False

# Function to update the GUI with the current emotion prediction and display detected face(s)
def process_emotion_video():
    ret, frame = cap.read()

    if process_video:
        if ret and not pause_video:  # Check if processing is not paused
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract and preprocess the detected face
                face_roi = frame_gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                face_roi = np.expand_dims(face_roi, axis=-1)
                face_roi = face_roi.astype('float32') / 255.0
                face = np.expand_dims(face_roi, axis=0)

                # Make a prediction on the preprocessed face
                prediction = model.predict(face)
                emotion_index = np.argmax(prediction)
                emotion_label = emotion_labels[emotion_index]

                # Display the emotion label on the frame
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame to RGB format for displaying in Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)

        # Update the GUI with the current frame
        video_label.config(image=frame_tk)
        video_label.image = frame_tk

    # Update the GUI every 100 milliseconds
    root.after(100, process_emotion_video)

# Function to handle quitting the application
def quit_application():
    response = messagebox.askyesno("Quit", "Are you sure you want to quit?")
    if response:
        cap.release()  # Release the VideoCapture
        cv2.destroyAllWindows()  # Close OpenCV windows
        root.destroy()

# Create the main GUI window
root = tk.Tk()
root.title("Human's Facial Emotion Detection")

# Set the background color to light blue
root.configure(bg='#ADD8E6')  # Light Blue

# Create a frame for the label and buttons (first frame)
frame = tk.Frame(root, bg='#ADD8E6')  # Light Blue
frame.pack(expand=True, fill="both")

# Create a label for the title with large font size
title_label_large = tk.Label(frame, text="Welcome to Human's Facial Emotion Detection", font=("Helvetica", 26, "bold"),
                             bg='#ADD8E6')  # Bold and Light Blue
title_label_large.pack(pady=20)

# Create a label for the title with normal font size (hidden initially)
title_label_normal = tk.Label(frame, text=" Facial Emotion Detecting", font=("Helvetica", 12),
                               bg='#ADD8E6')  # Normal font and Light Blue

# Create a label to display the video feed
video_label = tk.Label(frame)
video_label.pack()

# Create a frame for the buttons (first frame)
button_frame = tk.Frame(frame, bg='#ADD8E6')  # Light Blue
button_frame.pack(pady=10)

# Create a "Start" button (first frame)
start_button = tk.Button(button_frame, text="Start", command=start_video_processing, bg='#4CAF50',  # Green
                         fg='white', font=("Helvetica", 12))  # White text and Green button
start_button.pack(side="left", padx=20)

# Create a "Quit" button (first frame)
quit_button = tk.Button(button_frame, text="Quit", command=quit_application, bg='#FF0000',  # Red
                        fg='white', font=("Helvetica", 12))  # White text and Red button
quit_button.pack(side="left", padx=20)

# Create a frame for the buttons (second frame)
button_frame_second = tk.Frame(root, bg='#ADD8E6')  # Light Blue
button_frame_second.pack(expand=True, fill="both")

# Create a "Resume" button (second frame)
resume_button = tk.Button(button_frame_second, text="Resume", command=resume_video_processing, bg='blue',  # Blue
                           fg='white', font=("Helvetica", 12))  # White text and Blue button

# Create a "Pause" button (second frame)
pause_button = tk.Button(button_frame_second, text="Pause", command=pause_video_processing, bg='orange',  # Orange
                         fg='black', font=("Helvetica", 12))  # Black text and Orange button

# Create a "Quit" button (second frame)
quit_button_second = tk.Button(button_frame_second, text="Quit", command=quit_application, bg='#FF0000',  # Red
                              fg='white', font=("Helvetica", 12))  # White text and Red button

# Run the GUI main loop
root.mainloop()

# Release the VideoCapture when the GUI is closed
cap.release()
cv2.destroyAllWindows()
