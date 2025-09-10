# --- File: app.py ---
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import joblib
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('model/expression_model.h5')
label_map = joblib.load('model/label_map.pkl')
reverse_map = {v: k for k, v in label_map.items()}

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_normalized = roi_resized / 255.0
                roi_reshaped = roi_normalized.reshape(1, 48, 48, 1)
                prediction = model.predict(roi_reshaped)
                predicted_index = prediction.argmax()
                predicted_label = reverse_map[predicted_index]
                return render_template('result.html', prediction=predicted_label, image_url=filepath)

            return render_template('result.html', prediction="No face detected", image_url=filepath)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)