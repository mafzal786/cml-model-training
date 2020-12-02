import os
import numpy as np
from tensorflow.keras.models import load_model

# == For Testing ==
features = ['pixelarray']
args = {"pixelarray": "0,0,0,0,0,0,0.03137254901960784,0.5764705882352941,1,1,1,1,1,1,0.17254901960784313,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.011764705882352941,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.984313725490196,0.07450980392156863,0,0,0,0,0,0,0,0,0,0,1,1,1,0.6196078431372549,0,0,0,0,0,0,0,0,0,0,0,0.8980392156862745,1,1,1,1,0,0,0,0,0,0,0,0,0.48627450980392156,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.8470588235294118,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0784313725490196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.07058823529411765,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.7098039215686275,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.7568627450980392,1,0.7843137254901961,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.3176470588235294,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4470588235294118,1,1,1,0,0,0,0,0,0,0,0,0,0.8980392156862745,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.03137254901960784,0,0,0,0,0,0,0,0,0,0,0,0,0,0.06274509803921569,0.07058823529411765,0.07058823529411765,0.803921568627451,1,1,0.8941176470588236,0.07058823529411765,0.06666666666666667,0,0,0,0.0392156862745098,1,1,1,0.5098039215686274,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.9254901960784314,1,0.6901960784313725,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5254901960784314,1,0.07450980392156863,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.9254901960784314,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3803921568627451,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.24705882352941178,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.34509803921568627,1,0,0,0,0,0,0.9803921568627451,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0.3686274509803922,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0.9882352941176471,1,1,1,1,0.6784313725490196,0.08235294117647059,0,0,0,0,0,0.4235294117647059,1,1,1,1,1,1,1,0.3333333333333333,0,0,0,0,0,0,0,0,0,0.10588235294117647,0.8117647058823529,1,1,1,1,1,1,1,1,1,1,1,1,0.21568627450980393,0.12941176470588237,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"}



# == Main Function ==
def PredictFunc(args):
  
    
    
    data=list(args.values())[0]
    
    
    pixelarray = data
    arr_digit=pixelarray.split(',')
    arr32_digit = np.array(arr_digit,dtype='float32')

    model = load_model("model.h5")
    predict_img = arr32_digit.reshape(1,28,28,1)
    prediction = model.predict(predict_img)
    response = format(prediction.argmax())
    return response
