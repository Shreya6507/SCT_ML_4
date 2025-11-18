Hand Gesture Recognition (MediaPipe + sklearn)

This project provides a simple pipeline to collect hand-gesture samples using a webcam, train a classifier on MediaPipe hand landmarks, and run real-time inference for gesture-based control.

Key pieces:
- `src/data_collection.py` — capture labelled landmark samples from webcam and save to CSV
- `src/train_classifier.py` — train a classifier (RandomForest) on collected samples and save a model
- `src/infer_realtime.py` — run webcam inference with the saved model and show labels on-screen

This approach uses MediaPipe Hands to extract 21 hand landmarks per detected hand, then trains a lightweight classifier on these landmarks. It's fast and robust for common gestures.

Requirements and quick start are in `requirements.txt`.

Usage summary:
1. Install dependencies: see `requirements.txt`.
2. Collect samples for each gesture using `python src/data_collection.py --label thumbs_up`.
3. Train: `python src/train_classifier.py --data data/gestures.csv`.
4. Run real-time inference: `python src/infer_realtime.py --model models/gesture_clf.joblib`.

See the scripts for detailed flags and options.
