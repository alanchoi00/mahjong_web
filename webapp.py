from flask import Flask, render_template, request, send_from_directory, Response
from ultralytics import YOLO
import os
import cv2
import argparse

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECT_FOLDER = 'runs/detect'

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

tile_labels = [
  'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9',
  'sb1', 'sb2', 'sb3', 'sb4', 'sb5', 'sb6', 'sb7', 'sb8', 'sb9',
  'sc1', 'sc2', 'sc3', 'sc4', 'sc5', 'sc6', 'sc7', 'sc8', 'sc9',
  'hwe', 'hws', 'hww', 'hwn', 'hdr', 'hdg', 'hdw',
  'bs1', 'bs2', 'bs3', 'bs4',
  'bf1', 'bf2', 'bf3', 'bf4'
]

@app.route("/")
def index():
  return render_template('index.html')

def process_detections(detections):
  # Create a dictionary to store detection details
  results = {}

  for detection in detections:
    detected_indices = detection.boxes.cls.cpu().numpy().astype(int)
    detected_conf = detection.boxes.conf.cpu().numpy()
    detected_boxes = detection.boxes.xyxy.cpu().numpy()

    for idx, conf, box in zip(detected_indices, detected_conf, detected_boxes):
      label = detection.names[idx]
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

        results = process_detections(detections)
        print("Detection results:", results)
        subfolders = [f for f in os.listdir(DETECT_FOLDER) if os.path.isdir(os.path.join(DETECT_FOLDER, f))]
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(DETECT_FOLDER, x)))
        prediction_filename = os.listdir(os.path.join(DETECT_FOLDER, latest_subfolder))[0]

        return render_template('index.html', upload=True, filename=f.filename,
                               upload_path=f"/uploads/{f.filename}",
                               image_path=f"/detect/{latest_subfolder}/{prediction_filename}",
                               )

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
