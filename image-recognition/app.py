# app.py
print("Step 1: Script starting...") # <-- DEBUG

import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

print("Step 2: Imports successful.") # <-- DEBUG

# --- SETUP ---
app = Flask(__name__)

# Load the pre-trained MobileNetV2 model
print("Step 3: Loading the pre-trained model... (This may take a moment)") # <-- DEBUG
model = MobileNetV2(weights='imagenet')
print("Step 4: Model loaded successfully.") # <-- DEBUG


# --- HELPER FUNCTION ---
def model_predict(image_path):
    """
    Preprocesses an image and makes a prediction using the loaded model.
    """
    img = Image.open(image_path)
    img = img.resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image found", 400
    
    f = request.files['image']
    
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
        
    file_path = os.path.join('static/uploads', f.filename)
    f.save(file_path)
    
    preds = model_predict(file_path)
    
    decoded_preds = decode_predictions(preds, top=1)[0]
    
    top_prediction = decoded_preds[0]
    class_name = top_prediction[1]
    confidence = top_prediction[2]
    
    result_text = f"I am {confidence:.2%} confident this is a {class_name.replace('_', ' ')}."
    
    return render_template('result.html', prediction_text=result_text, image_name='uploads/' + f.filename)


if __name__ == '__main__':
    print("Step 5: Starting the Flask server...") # <-- DEBUG
    app.run(debug=True)