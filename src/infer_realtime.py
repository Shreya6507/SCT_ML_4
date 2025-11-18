"""Run real-time gesture inference using saved model and MediaPipe.

Example:
  python src/infer_realtime.py --model models/gesture_clf.joblib
"""
import argparse
from collections import deque
import joblib
import cv2
import numpy as np
import mediapipe as mp

from utils import landmarks_to_feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/gesture_clf.joblib')
    parser.add_argument('--smooth', type=int, default=5, help='Smoothing window for predicted labels')
    args = parser.parse_args()

    data = joblib.load(args.model)
    clf = data['model']
    le = data['le']

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    label_window = deque(maxlen=args.smooth)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            display_text = 'No hand'
            prob = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = landmarks_to_feature(hand_landmarks.landmark)
                X = np.array(features).reshape(1, -1)
                probs = clf.predict_proba(X)[0]
                idx = probs.argmax()
                display_text = le.inverse_transform([idx])[0]
                prob = probs[idx]
                label_window.append(display_text)
            else:
                # keep previous labels if any
                if label_window:
                    label_window.append(label_window[-1])

            # smoothing
            if label_window:
                # most common in window
                vals, counts = np.unique(np.array(label_window), return_counts=True)
                display_text = vals[counts.argmax()]

            cv2.putText(img, f'{display_text} {prob:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imshow('Gesture Recognition', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
