"""Recognize static ASL letters A-Z and digits 0-9 with a webcam."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sign_features import landmarks_to_features

ROOT = Path(__file__).resolve().parent
GESTURE_MODEL = ROOT / "models" / "gesture_recognizer.task"
CLASSIFIER_MODEL = ROOT / "models" / "asl_signs.pkl"


def main() -> None:
    if not CLASSIFIER_MODEL.is_file():
        raise FileNotFoundError("Train the classifier first: py train_asl_digits.py <dataset-folder>")
    with CLASSIFIER_MODEL.open("rb") as file:
        classifier = pickle.load(file)["classifier"]

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
    )
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("Webcam is unavailable")

    timestamp = 0
    recognizer = vision.GestureRecognizer.create_from_options(options)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = max(time.monotonic_ns() // 1_000_000, timestamp + 1)
            result = recognizer.recognize_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
            vector = landmarks_to_features(result)
            text = "Show a hand"
            if vector is not None:
                probabilities = classifier.predict_proba([vector])[0]
                best = probabilities.argmax()
                text = f"ASL sign: {classifier.classes_[best]} ({probabilities[best]:.0%})"
            cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Esc: quit", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("ASL static signs: A-Z, 0-9", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        recognizer.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
