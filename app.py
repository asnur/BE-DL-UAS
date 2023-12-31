from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.models import load_model
from keras.applications import inception_v3

# Model path
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

# Load test dataset with labels
test_data_path = './data/test_image'


#Define image parameters
img_width, img_height = 600, 400

#Prediction Function
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  x = inception_v3.preprocess_input(x)

  result = model.predict(x)

  prediction = ''

  answer = np.argmax(result[0])
  if answer == 0:
    prediction = "Cloudy"
    print("Predicted: Cloudy")
  elif answer == 1:
    prediction = "Rainy"
    print("Predicted: Rainy")
  elif answer == 2:
    prediction = "Shine"
    print("Predicted: Shine")
  elif answer == 3:
    prediction = "Sunrise"
    print("Predicted: Sunrise")

  accuracy = result[0][answer] * 100
  print("Accuracy: ", accuracy)

  return prediction, accuracy

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT

#Flask
app = Flask(__name__)

#CORS
CORS(app)

#Define route for the API
@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['files']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('data/uploaded', filename)
            file.save(file_path)
            result, accuracy = predict(file_path)
            
            return jsonify({'result': result, 'accuracy': accuracy})
    return "Invalid Request"

if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0",port=6969)