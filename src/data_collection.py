"""Collect hand landmark samples using MediaPipe and save to CSV.

Usage examples:
  python src/data_collection.py --label thumbs_up --out data/gestures.csv --max-samples 200

Press 's' to save the current detected hand landmarks as one sample.
Press 'q' to quit.
"""
import argparse
import csv
import os
import sys
from time import time

import cv2
import mediapipe as mp
import numpy as np

from utils import ensure_dir, landmarks_to_feature


def ensure_header(path, n_landmarks=21):
    if not os.path.exists(path):
        ensure_dir(os.path.dirname(path) or '.')
        header = []
        for i in range(n_landmarks):
            header += [f'x{i}', f'y{i}', f'z{i}']
        header += ['label']
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True, help='Label for this recording (e.g. thumbs_up)')
    parser.add_argument('--out', default='data/gestures.csv', help='CSV file to append samples to')
    parser.add_argument('--max-samples', type=int, default=0, help='Stop after capturing this many samples (0 = unlimited)')
    args = parser.parse_args()

    ensure_header(args.out)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    saved = 0
    last_save = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('No camera frame. Exiting.')
                break

            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert landmarks to normalized list
                features = landmarks_to_feature(hand_landmarks.landmark)
                text = f"Detected - press 's' to save ({saved} saved)"
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(img, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow('Data Collection - press s to save, q to quit', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and results.multi_hand_landmarks:
                # throttle saves (small debounce)
                if time() - last_save < 0.15:
                    continue
                last_save = time()
                # write feature + label to csv
                with open(args.out, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(features + [args.label])
                saved += 1
                print(f'Saved sample #{saved} for label {args.label}')
                if args.max_samples and saved >= args.max_samples:
                    print('Reached max samples. Exiting.')
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
