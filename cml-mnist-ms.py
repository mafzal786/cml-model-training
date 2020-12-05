import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('draw.html')

@app.route("/predict")
def predict():
    pixelarray = request.args.get('pixelarray')
    
    arr_digit=pixelarray.split(',')
    arr32_digit = np.array(arr_digit,dtype='float32')

    model = load_model("model.h5")
    predict_img = arr32_digit.reshape(1,28,28,1)
    prediction = model.predict(predict_img)
    response = "<h1>Your handwritten digit is: {}</h1>".format(prediction.argmax())
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT",8080))
    app.run(host='127.0.0.1', port=int(os.environ['CDSW_APP_PORT']))
