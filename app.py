import os
import cv2
import base64
import numpy as np
from keras._tf_keras.keras.models import load_model
from flask import Flask,render_template,request

app = Flask(__name__,template_folder="templates",static_folder="static")

model = load_model('models/MobileNet.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html')

    img_to_read = file.read()  # Get the content of the file
    nparr = np.frombuffer(img_to_read, np.uint8)  # Convert the content to a NumPy array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Read the image from the array
    print(img.shape)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res_img = cv2.resize(img,(224,224))
    print(res_img.shape)
    X = res_img/255
    pred = model.predict(X.reshape(1,224,224,3))
    prediction = np.argmax(pred)
    if prediction==1:
        result = 'The image is of a dog'
    else:
        result = 'The image is of a cat'

    _, img_encoded = cv2.imencode('.png', res_img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')


    return render_template('index.html',prediction=result,image=img_base64)


if __name__ == '__main__':
     app.run(debug=True)