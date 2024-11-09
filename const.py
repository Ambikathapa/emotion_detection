import numpy as np

CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
   
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))