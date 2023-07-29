from flask import Flask, request, jsonify
import onnxruntime as ort
import cv2 
import numpy as np
from tools.process_face import *

# Create the Flask app
app = Flask(__name__)

# Load the face detection model
face_detect_model = ort.InferenceSession("models/version-RFB-320.onnx")
threshold = 0.7

def preprocess_image(image_data):
    # Convert image_data to a NumPy array
    image_data_np = np.frombuffer(image_data, np.uint8)
    # Decode the image data
    image = cv2.imdecode(image_data_np, cv2.IMREAD_COLOR)
    # Preprocess the image
    image_preprocess = preprocess(image)
    
    return image_preprocess, image

def run_face_detection(image_preprocess, image_source):
    input_name = face_detect_model.get_inputs()[0].name
    confidences, boxes = face_detect_model.run(None, {input_name: image_preprocess})
    boxes, labels, probs = post_process(image_source.shape[1], image_source.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs

# Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
    # Get the input data
        input_data = request.files["image"].read()
        
        # Preprocess the input image
        image_preprocess, image_source = preprocess_image(input_data)
        
        # # Run the face detection inference
        boxes, labels, probs = run_face_detection(image_preprocess,image_source)
        outputs = {
            "boxes": boxes, 
            "labels": labels, 
            "probs": probs
        }
        return jsonify(outputs)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
