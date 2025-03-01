from flask import Flask, render_template, request, redirect,jsonify,url_for,redirect,session
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from chatModel import ask_question_with_sources

app = Flask(__name__)
# Load your models
misleading_model = load_model('misleading_cnnmodel.h5')
alteration_model = load_model(
    'alteration_model.h5')
cnn_model = load_model('cnnmodel.h5')

def get_model(model_name):
    if model_name == 'misleading_cnnmodel':
        return misleading_model
    elif model_name == 'alteration_model':
        return alteration_model
    elif model_name == 'cnnmodel':
        return cnn_model
    else:
        # Handle unknown model names
        return None

# Define a function to preprocess the image before prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def preprocess_image_altered(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def preprocess_image_misleading(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def process_prediction(file, selected_model):
    # Load the selected model
    model = get_model(selected_model)
    if model is None:
        return None

    # Choose the appropriate preprocessing function based on the selected model
    if selected_model == 'misleading_cnnmodel':
        preprocess_function = preprocess_image_misleading
    elif selected_model == 'alteration_model':
        preprocess_function = preprocess_image_altered
    elif selected_model == 'cnnmodel':
        preprocess_function = preprocess_image
    else:
        # Handle other models as needed
        return None

    # Save the uploaded image to a temporary folder
    temp_path = os.path.join('static', 'temp', file.filename)
    file.save(temp_path)

    # Preprocess the image using the selected preprocessing function
    img_array = preprocess_function(temp_path)

    try:
        # Make prediction
        prediction = model.predict(img_array)[0]
        # Uncomment the following line if you want to see the raw prediction values
        # print("Raw Prediction Values:", prediction)

        # Get the predicted class (0 or 1)
        predicted_class = prediction[0]

        #print("Prediction Successful. Predicted Class:", predicted_class)
        return predicted_class

    except Exception as e:
        # Print the exception for debugging
        #print("Prediction Error:", e)
        return None

@app.route('/')
def index():
    return render_template('index_project.html')

@app.route('/predict', methods=['POST'])
def predict():
    file_model1 = request.files.get('file_model1')
    file_model2 = request.files.get('file_model2')
    file_model3 = request.files.get('file_model3')

    if file_model1:
        file = file_model1
        selected_model = 'misleading_cnnmodel'
    elif file_model2:
        file = file_model2
        selected_model = 'alteration_model'
    elif file_model3:
        file = file_model3
        selected_model = 'cnnmodel'
    else:
        return render_template('index_project.html', error='No file selected')

    # Process prediction
    predicted_class = process_prediction(file, selected_model)
    

    if predicted_class is None:
        return render_template('index_project.html', error='Invalid model selection')
    
    print("Predicted Class (in predict function):", predicted_class)
    rounded_predicted_class = round(predicted_class)
    print("rounded_predicted_class=",rounded_predicted_class)
    # Render the modified index template with the predicted class, filename, and model information
    return render_template('index_project_modified.html', predicted_class=rounded_predicted_class, filename=file.filename, selected_model=selected_model)

@app.route("/chat",methods=['POST'])
def  chat():
    try:
        text=request.get_json().get("message")
        response=ask_question_with_sources(text)
        message={"answer":response}
        return jsonify(message)
    except Exception as e:
        return jsonify({"answer": {str(e)}})


if __name__ == '__main__':
    app.run(debug=True)