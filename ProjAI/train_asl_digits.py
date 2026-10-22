"""Train a static ASL digit (0-9) classifier from a folder image dataset."""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sign_features import landmarks_to_features

ROOT = Path(__file__).resolve().parent
GESTURE_MODEL = ROOT / "models" / "gesture_recognizer.task"
LABELS = [str(value) for value in range(10)]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_label(path: Path) -> str | None:
    """Find a digit directory anywhere below the dataset directory."""
    for part in reversed(path.parts[:-1]):
        if part in LABELS:
            return part
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path,
                        help="Extracted ASL-HG directory containing folders 0 through 9")
    parser.add_argument("--per-class", type=int, default=400,
                        help="Maximum images per digit (default: 400)")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "asl_digits.pkl")
    args = parser.parse_args()

    if not GESTURE_MODEL.is_file():
        raise FileNotFoundError(f"Missing MediaPipe model: {GESTURE_MODEL}")
    if not args.dataset.is_dir():
        raise NotADirectoryError(args.dataset)

    files_by_label = {label: [] for label in LABELS}
    for path in args.dataset.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            label = image_label(path)
            if label is not None and len(files_by_label[label]) < args.per_class:
                files_by_label[label].append(path)
    missing = [label for label, files in files_by_label.items() if not files]
    if missing:
        raise ValueError(f"No images found for digits: {', '.join(missing)}")

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    features: list[np.ndarray] = []
    targets: list[str] = []
    skipped = Counter()
    recognizer = vision.GestureRecognizer.create_from_options(options)
    try:
        for label, paths in files_by_label.items():
            for number, path in enumerate(paths, start=1):
                image = cv2.imread(str(path))
                if image is None:
                    skipped[label] += 1
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                vector = landmarks_to_features(result)
                if vector is None:
                    skipped[label] += 1
                    continue
                features.append(vector)
                targets.append(label)
                if number % 100 == 0:
                    print(f"{label}: {number}/{len(paths)}")
    finally:
        recognizer.close()

    print("Usable images:", Counter(targets), "; skipped:", dict(skipped))
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(features), np.array(targets), test_size=0.2,
        random_state=42, stratify=targets,
    )
    classifier = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=42,
    )
    classifier.fit(x_train, y_train)
    print(classification_report(y_test, classifier.predict(x_test), digits=3))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump({"classifier": classifier, "labels": LABELS}, file)
    print(f"Saved model: {args.output}")


if __name__ == "__main__":
    main()
