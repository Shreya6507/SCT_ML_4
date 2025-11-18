from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
from flask import send_from_directory

app = Flask(__name__)

# Default model path
MODEL_PATH = os.environ.get('GESTURE_MODEL', 'models/gesture_clf.joblib')


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    data = joblib.load(path)
    return data['model'], data['le']


_MODEL = None


def get_model():
    """Lazy-load the model on first request to keep startup light."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


@app.route('/')
def index():
    # Serve the static demo page
    return app.send_static_file('index.html')


@app.route('/status', methods=['GET'])
def status():
    model_pair = get_model()
    ok = model_pair is not None
    return jsonify({'ready': ok, 'model_path': MODEL_PATH if ok else None})


def parse_landmarks(payload):
    """Accepts either a flat list or list of {x,y,z} dicts, returns flat list of floats."""
    if isinstance(payload, dict) and 'landmarks' in payload:
        landmarks = payload['landmarks']
    else:
        landmarks = payload

    # If landmarks are objects/dicts with x/y/z
    if len(landmarks) and isinstance(landmarks[0], dict):
        flat = []
        for lm in landmarks:
            flat.extend([float(lm.get('x', 0)), float(lm.get('y', 0)), float(lm.get('z', 0))])
        return flat

    # otherwise assume flat numeric list
    return [float(x) for x in landmarks]


@app.route('/predict', methods=['POST'])
def predict():
    global MODEL
    model_pair = get_model()
    if model_pair is None:
        return jsonify({'error': 'Model not loaded. Train or provide a model at models/gesture_clf.joblib'}), 400

    payload = request.get_json(force=True)
    try:
        feat = parse_landmarks(payload)
    except Exception as e:
        return jsonify({'error': 'Invalid payload: ' + str(e)}), 400

    X = np.array(feat).reshape(1, -1)

    clf, le = model_pair
    # Align input size to what the classifier expects. Some saved models may have been
    # trained with a different number of feature columns (CSV issues or dropped NaN cols).
    n_expected = getattr(clf, 'n_features_in_', None)
    if n_expected is not None:
        if X.shape[1] < n_expected:
            # pad with zeros
            pad = np.zeros((1, n_expected - X.shape[1]))
            X = np.concatenate([X, pad], axis=1)
        elif X.shape[1] > n_expected:
            # trim extra features
            X = X[:, :n_expected]
    else:
        # If estimator does not expose n_features_in_, do a basic sanity check
        if X.shape[1] < 21*3:
            return jsonify({'error': f'Expected at least {21*3} features, got {X.shape[1]}'}), 400
    try:
        probs = clf.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        label = le.inverse_transform([idx])[0]
        prob = float(probs[idx])
    except Exception:
        # Some sklearn classifiers may not implement predict_proba
        pred = clf.predict(X)[0]
        label = le.inverse_transform([int(pred)])[0]
        prob = None

    return jsonify({'label': label, 'probability': prob})


@app.route('/train', methods=['POST'])
def train():
    """Trigger retraining using existing `data/gestures.csv` and save the model.
    This runs the project's `src/train_classifier.py` script as a subprocess.
    """
    import subprocess, sys

    script = os.path.join('src', 'train_classifier.py')
    if not os.path.exists(script):
        return jsonify({'error': 'Training script not found'}), 500

    # Optional params: data path and out path
    data_path = request.json.get('data', 'data/gestures.csv') if request.is_json else 'data/gestures.csv'
    out_path = request.json.get('out', 'models/gesture_clf.joblib') if request.is_json else 'models/gesture_clf.joblib'

    cmd = [sys.executable, script, '--data', data_path, '--out', out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        return jsonify({'ok': False, 'stdout': proc.stdout, 'stderr': proc.stderr}), 500

    # reload model
    try:
        # Clear cache and reload
        global _MODEL
        _MODEL = None
        _MODEL = load_model(out_path)
    except Exception:
        pass

    return jsonify({'ok': True, 'stdout': proc.stdout, 'stderr': proc.stderr})


if __name__ == '__main__':
    # Run on localhost:5000
    # Use non-debug mode and disable the reloader to reduce overhead.
    # If you want a production server on Windows, consider using 'waitress' and
    # setting the environment variable USE_WAITRESS=1 before starting.
    import os
    use_waitress = os.environ.get('USE_WAITRESS') == '1'
    if use_waitress:
        try:
            from waitress import serve
            serve(app, host='127.0.0.1', port=5000)
        except Exception:
            # fallback to Flask dev server if waitress not available
            app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    else:
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
