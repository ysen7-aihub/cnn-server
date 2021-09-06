from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

cnnModel = load_model('image_model.h5')
labels = [0,1,2,3,4]
# 0:rain, 1:foggy, 2:cloudy, 3:shine, 4:sunrise

def preprocessing(inputImg):
    img = Image.open(inputImg)
    img = img.resize((150,150))
    pixel = image.img_to_array(img)
    pixel = np.expand_dims(pixel, axis=0)

    return pixel

def create_result(x):
    result = np.where(x==0, 0.5,
                      np.where(x==1, 0.4,
                               np.where(x==2, 0.2,
                                        np.where(x==3, 0.7,
                                                 np.where(x==4,1,0.5)))))
                      
    return result

@app.route('/predict/img', methods=['POST'])
def predict():
    inputImg = request.files['file'] #BytesIO object
    
    pixel = preprocessing(inputImg)
    preds = cnnModel.predict(pixel)
    
    pred = np.argmax(preds)
    res = labels[pred]
    #print('label: ',res)

    result = create_result(res)
    result = float(result)
    return jsonify({"score2" : result})

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)