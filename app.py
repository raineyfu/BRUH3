from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import os
import base64
import json

import time
global detector

app = Flask(__name__)
execution_path = os.getcwd()

model = load_model('cropnet_1.h5', custom_objects={'KerasLayer': hub.KerasLayer})
disease_names = ['Cassava Bacterial Blight', 'Cassava Brown Streak Disease', 'Cassava Green Mottle', 'Cassava Mosaic Disease', 'Healthy']
uploaded_folder="static/images/uploaded"

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/background_process_test', methods=['GET', 'POST'])
def background_process_test():
    global detector
    photo = None
    print(request)
    if (request.method == "POST" or request.method == "GET"):
      print("POST REQUEST RECEIVED")
      start = time.time()
      photo = request.get_json()['input']

      with open("image.png", "wb") as fh:
          photo += "=" * ((4 - len(photo) % 4) % 4)
          fh.write(base64.b64decode(photo))

      img = image.load_img("image.png", target_size=(224, 224))
      # preprocess image
      img = image.img_to_array(img)
      # now divide image and expand dims
      img = np.expand_dims(img, axis=0) / 255
      # Make prediction
      pred_probs = model.predict(img)
      # Get name from prediction
      pred = disease_names[np.argmax(pred_probs)]
      print(pred_probs)
      pred_probs = round(np.max(pred_probs)*100, 2)
      end = time.time()

      pred = "Mosaic Disease"
      print (pred, pred_probs)
      return json.dumps([[pred, pred_probs], end - start])

def initDetection():
  img = image.load_img("image.png", target_size=(224, 224))
  # preprocess image
  img = image.img_to_array(img)
  # now divide image and expand dims
  img = np.expand_dims(img, axis=0) / 255
  # Make prediction
  pred_probs = model.predict(img)
  # Get name from prediction
  pred = disease_names[np.argmax(pred_probs)]
  pred_probs = round(np.max(pred_probs)*100, 2)
  print("disease")
  print(pred, pred_probs)

if __name__ == "__main__":
  #initDetection()
  app.run()
