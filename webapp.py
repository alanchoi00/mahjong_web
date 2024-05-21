from flask import Flask, render_template, request, send_from_directory, Response
from ultralytics import YOLO
import os
import cv2
import argparse
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECT_FOLDER = 'runs/detect'

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

@app.route("/")
def hello_world():
  return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
  if request.method == "POST":
    if 'file' in request.files:
      f = request.files['file']
      basepath = os.path.dirname(__file__)
      filepath = os.path.join(basepath, UPLOAD_FOLDER, f.filename)
      print("Upload folder is ", filepath)
      f.save(filepath)

      file_extension = f.filename.rsplit('.', 1)[1].lower()

      if file_extension == 'jpg':
        img = cv2.imread(filepath)
        model = YOLO('yolov9c.pt')
        detections = model(img, save=True)
        folder_path = DETECT_FOLDER
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
        prediction_filename = os.listdir(os.path.join(folder_path, latest_subfolder))[0]

        return render_template('index.html', upload=True, filename=f.filename, upload_path=f"/uploads/{f.filename}", image_path=f"/detect/{latest_subfolder}/{prediction_filename}")

    print('No File Provided')
    return render_template('index.html', error="No file provided.")

  return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/detect/<folder>/<filename>')
def detect_file(folder, filename):
  directory = os.path.join(DETECT_FOLDER, folder)
  return send_from_directory(directory, filename)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Flask app exposing YOLOv9 models")
  parser.add_argument("--port", default=5000, type=int, help="port number")
  args = parser.parse_args()
  model = YOLO('yolov9c.pt')
  app.run(host="0.0.0.0", port=args.port)
