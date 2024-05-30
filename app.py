from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('models/best_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(data):
    data.fillna(method='ffill', inplace=True)
    return data

def encode_categorical_features(data):
    return pd.get_dummies(data, drop_first=True)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def add_features(data):
    data['car_age'] = 2024 - data['year']
    data.drop('year', axis=1, inplace=True)
    return data

def preprocess_data(data):
    data = handle_missing_values(data)
    data = encode_categorical_features(data)
    data = add_features(data)
    return data

def detect_damages(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    damage_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            damage_detected = True
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    
    cv2.imwrite(image_path, image)  # Save the image with detected damages

    return damage_detected

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        car_data = {
            'year': int(request.form['year']),
            'mileage': int(request.form['mileage']),
            'engine_health': int(request.form['engine_health']),
            'brand': request.form['brand'],
            'model': request.form['model'],
            'condition': request.form['condition']
        }

        df = pd.DataFrame([car_data])
        df = preprocess_data(df)
        df_scaled = scaler.transform(df)

        prediction = model.predict(df_scaled)
        predicted_price = prediction[0]

        return render_template('result.html', predicted_price=predicted_price)

# Upload route for damage detection
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        damage_detected = detect_damages(filepath)
        return render_template('result.html', damage_detected=damage_detected)

if __name__ == '__main__':
    app.run(debug=True)
