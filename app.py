from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your machine learning models
naive_bayes_model = joblib.load('models/recommendation_model.pkl')
plant_disease_model = tf.keras.models.load_model('models\plant_disease_detection.h5')

labels = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy',
    4: 'Not a plant', 5: 'Blueberry___healthy', 6: 'Cherry___Powdery_mildew', 7: 'Cherry___healthy',
    8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 9: 'Corn___Common_rust', 10: 'Corn___Northern_Leaf_Blight',
    11: 'Corn___healthy', 12: 'Grape___Black_rot', 13: 'Grape___Esca_(Black_Measles)',
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 15: 'Grape___healthy', 16: 'Orange___Haunglongbing_(Citrus_greening)',
    17: 'Peach___Bacterial_spot', 18: 'Peach___healthy', 19: 'Pepper,_bell___Bacterial_spot',
    20: 'Pepper,_bell___healthy', 21: 'Potato___Early_blight', 22: 'Potato___Late_blight', 23: 'Potato___healthy',
    24: 'Raspberry___healthy', 25: 'Soybean___healthy', 26: 'Squash___Powdery_mildew',
    27: 'Strawberry___Leaf_scorch', 28: 'Strawberry___healthy', 29: 'Tomato___Bacterial_spot',
    30: 'Tomato___Early_blight', 31: 'Tomato___Late_blight', 32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot', 34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot', 36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    37: 'Tomato___Tomato_mosaic_virus', 38: 'Tomato___healthy'
}

imgSize = 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

@app.route('/plant_disease')
def plant_disease():
    return render_template('plant_disease.html')

@app.route('/crop_predict', methods=['POST'])
def crop_predict():
    # Retrieve form data from the POST request
    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potassium = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Create a DataFrame from the form inputs
    user_input = pd.DataFrame({
        'N': [nitrogen],
        'P': [phosphorus],
        'K': [potassium],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Predict using the Naive Bayes model
    prediction = naive_bayes_model.predict(user_input)
    crop_recommendation = prediction[0]  # Assuming prediction returns a single value

    # Send the result back as JSON to be handled by JavaScript on the same page
    return jsonify({'prediction': crop_recommendation})

@app.route('/disease_predict', methods=['POST'])
def disease_predict():
    # Retrieve the uploaded image from the request
    file = request.files['image']
    img = Image.open(file)

    # Preprocess the image
    img = img.resize((imgSize, imgSize))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = img_array.reshape(-1, imgSize, imgSize, 3)  # Reshape for the model

    # Predict using the plant disease model
    prediction = plant_disease_model.predict(img_array)
    predicted_class = np.argmax(prediction[0])  # Get the index of the highest probability
    predicted_label = labels[predicted_class]

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
