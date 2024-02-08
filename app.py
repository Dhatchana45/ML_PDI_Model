from flask import Flask, render_template, request,jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging

app = Flask(__name__, template_folder='templates')
# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Load your pre-trained model using h5 file (replace 'model.h5' with your actual model file)
model_path = 'trained.h5'
model = load_model(model_path)

# Function to preprocess user-uploaded image for prediction
def preprocess_user_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, 224, 224, 3))
    img_array = img_array / 255.0
    return img_array

# Function to perform prediction
def perform_prediction(user_uploaded_image):
    # Save the user-uploaded image to a temporary file
    temp_image_path = "temp_user_image.jpg"
    user_uploaded_image.save(temp_image_path)

    # Preprocess the user-uploaded image
    processed_image = preprocess_user_image(temp_image_path)

    # Perform prediction using the loaded model
    predictions = model.predict(processed_image)

    # Assuming the model returns a one-hot encoded array, extract the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Return the result as a dictionary
    result = {'predicted_class_index': predicted_class_index, 'confidence': float(predictions[0][predicted_class_index])}
    return result

# Function to get the disease name based on the predicted class index
def get_disease_name(predicted_class_index):
    # Define your mapping here
    disease_mapping = {
        0: 'Bacterial infection',
        1: 'Contact allergies',
        2: 'Folliculitis',
        3: 'Fungal infection',
        4: 'Impetigo',
        5: 'Parasite allergies',
        6: 'Pyoderma'
        # Add more as needed
    }

    # Return the corresponding disease name
    return disease_mapping.get(predicted_class_index, 'Unknown Disease')
@app.route('/save_pet-details',methods=['POST'])
def svae_pet_details():
    data=request.get_json()
    name=data['name']
    mobile_number=data['MobileNumber']
    dog_breed=data['dogbreed']
    age_of_dog=data['ageofdog']
    print(f"Recieved details:{name},{mobile_number},{dog_breed},{age_of_dog}")
    logging.info(f"Received details: {name}, {mobile_number}, {dog_breed}, {age_of_dog}")

    # Return a response to the client
    return jsonify({'status': 'success'})
# Your existing /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user-uploaded image from the form
    user_uploaded_image = request.files['file']

    # Perform the prediction using your model
    prediction_result = perform_prediction(user_uploaded_image)

    # Render the results.html template with the prediction values
    return render_template('result.html', prediction=prediction_result, get_disease_name=get_disease_name)

# Define a route for the root URL ("/")
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    app.run(host=host, port=port, debug=True)

    # Print the URL of the web application
print(f"Web application running at http://{host}:{port}/")


