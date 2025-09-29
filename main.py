# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# YOLOv8 Models
models = {
    "Brain Tumor MRI": YOLO("models/brain_tumor_mri.pt"),  # names: tumor
    "Brain Tumor CT": YOLO("models/brain_tumor_ct.pt"),  # names: tumor
    "Pneumonia CT": YOLO("models/pneumonia_ct.pt"),  # names: Pneumonia, covid
    "Pneumonia X-ray": YOLO("models/pneumonia_xray.pt"),  # names: pneumonia
    "Breast Cancer Mammogram": YOLO("models/breast_cancer_mammogram.pt"),  # names: cancer, normal
    "Breast Cancer Ultrasound": YOLO("models/breast_cancer_ultrasound.pt")  # names: benign, malignant, normal
}

# Disease groups for simultaneous diagnosis
disease_groups = {
    "Brain Tumor": ["MRI", "CT"],
    "Pneumonia": ["CT", "X-ray"],
    "Breast Cancer": ["Mammogram", "Ultrasound"]
}


@app.route('/')
def home():
    return render_template('index.html', diseases=disease_groups.keys())


@app.route('/get_test_types/<disease>')
def get_test_types(disease):
    return jsonify({
        'test_types': disease_groups[disease],
        'can_upload_both': len(disease_groups[disease]) > 1
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file1' not in request.files and 'file2' not in request.files:
        return jsonify({'error': 'No files uploaded'})

    disease = request.form['disease']
    test_types = disease_groups[disease]
    results = []
    file_paths = []

    for i, test_type in enumerate(test_types):
        file_key = f'file{i + 1}'
        if file_key not in request.files:
            continue

        file = request.files[file_key]
        if file.filename == '':
            continue

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_paths.append(filepath)

            model_key = f"{disease} {test_type}"
            severity, stage = diagnose_disease(filepath, disease, test_type)

            results.append({
                'test_type': test_type,
                'severity': severity,
                'stage': stage,
                'image_path': filepath
            })

    if not results:
        return jsonify({'error': 'No valid files uploaded'})

    # Calculate combined results if multiple scans
    combined_severity = None
    combined_stage = None

    if len(results) > 1:
        avg_severity = sum(r['severity'] for r in results) / len(results)
        combined_severity = round(avg_severity, 2)

        # Determine combined stage based on average severity
        combined_stage = "Mild" if combined_severity < 30 else "Moderate" if combined_severity < 70 else "Severe"

    return jsonify({
        'individual_results': results,
        'combined_severity': combined_severity,
        'combined_stage': combined_stage
    })


def diagnose_disease(image_path, disease, test_type):
    model_key = f"{disease} {test_type}"
    model = models[model_key]

    img = cv2.imread(image_path)
    results = model.predict(img)

    total_area = 0
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            total_area += (x2 - x1) * (y2 - y1)

    img_area = img.shape[0] * img.shape[1]
    severity = (total_area / img_area) * 100
    stage = "Mild" if severity < 30 else "Moderate" if severity < 70 else "Severe"

    return round(severity, 2), stage


if __name__ == '__main__':
    app.run(debug=True)
