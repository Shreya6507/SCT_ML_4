"""Train a classifier on collected landmark CSV and save model and label encoder.

Example:
  python src/train_classifier.py --data data/gestures.csv --out models/gesture_clf.joblib
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/gestures.csv', help='CSV with samples')
    parser.add_argument('--out', default='models/gesture_clf.joblib', help='Output model path')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f'Data file not found: {args.data}')

    df = pd.read_csv(args.data, header=0)

    # Allow CSVs that may not have the final column named 'label'.
    # If there is an explicit 'label' column, use it; otherwise assume the last column is the label.
    if 'label' in df.columns:
        y = df['label'].astype(str).values
        X_df = df.drop(columns=['label'])
    else:
        # fallback: last column is label
        y = df.iloc[:, -1].astype(str).values
        X_df = df.iloc[:, :-1]

    # Coerce feature columns to numeric (any non-numeric will become NaN), then fill or drop NaNs.
    X = X_df.apply(pd.to_numeric, errors='coerce')
    # If any columns are completely NaN (malformed CSV rows), drop them.
    X = X.dropna(axis=1, how='all')
    # Fill remaining NaNs with column mean (or 0 if column all-NaN)
    X = X.fillna(X.mean()).fillna(0)
    X = X.values

    # Drop rows with missing/empty labels which may have come from malformed CSV
    y_series = pd.Series(y)
    mask = ~y_series.isna() & (y_series.astype(str).str.strip() != '')
    X = X[mask.values]
    y_clean = y_series[mask].astype(str).values

    if len(set(y_clean)) < 2:
        raise ValueError('Need at least two classes with non-empty labels to train.')

    le = LabelEncoder()
    y_enc = le.fit_transform(y_clean)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=args.test_size, random_state=42, stratify=y_enc)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'Validation accuracy: {acc:.4f}')
    print(classification_report(y_test, preds, target_names=le.classes_))

    ensure_dir(os.path.dirname(args.out) or '.')
    joblib.dump({'model': clf, 'le': le}, args.out)
    print(f'Model saved to {args.out}')


if __name__ == '__main__':
    main()
