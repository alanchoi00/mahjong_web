from flask import Flask, render_template, request, send_from_directory, jsonify
from ultralytics import YOLO
import os
import cv2
import argparse

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECT_FOLDER = 'runs/detect'

# Ensure the upload and detect folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

@app.route("/")
def index():
  return render_template('index.html')

def process_detections(detections):
  conf_threshold = 0.6
  results = {}

  for detection in detections:
    detected_indices = detection.boxes.cls.cpu().numpy().astype(int)
    detected_conf = detection.boxes.conf.cpu().numpy()
    detected_boxes = detection.boxes.xyxy.cpu().numpy()

    for idx, conf, box in zip(detected_indices, detected_conf, detected_boxes):
      label = detection.names[idx]
      if conf > conf_threshold:
        if label not in results:
          results[label] = {
            "freq": 0,
            "conf": [],
            "boxes": []
          }
        results[label]["freq"] += 1
        results[label]["conf"].append(conf.tolist())
        results[label]["boxes"].append(box.tolist())
  return results

@app.route("/upload", methods=["POST"])
def upload_img():
  if 'upload_file' in request.files:
    f = request.files['upload_file']
    filepath = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(filepath)
    return jsonify({'filename': f.filename, 'upload_path': f"/uploads/{f.filename}"})
  return jsonify({'error': 'No file provided.'}), 400

@app.route("/predict/<filename>", methods=["GET"])
def predict_img(filename):
  filepath = os.path.join(UPLOAD_FOLDER, filename)
  img = cv2.imread(filepath)
  model = YOLO('yolov9c.pt')
  detections = model(img, save=True)

  results = process_detections(detections)
  subfolders = [f for f in os.listdir(DETECT_FOLDER) if os.path.isdir(os.path.join(DETECT_FOLDER, f))]
  latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(DETECT_FOLDER, x)))
  prediction_filename = os.listdir(os.path.join(DETECT_FOLDER, latest_subfolder))[0]

  return jsonify({
    'prediction_path': f"/detect/{latest_subfolder}/{prediction_filename}",
    'results': results
  })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/detect/<folder>/<filename>')
def detect_file(folder, filename):
  directory = os.path.join(DETECT_FOLDER, folder)
  return send_from_directory(directory, filename)

if __name__ == "__main__":
  # parser = argparse.ArgumentParser(description="Flask app exposing YOLOv9 models")
  # parser.add_argument("--port", default=5000, type=int, help="port number")
  # args = parser.parse_args()
  port = int(os.environ.get("PORT", 5000))
  app.run(debug=True, host="0.0.0.0", port=port)
