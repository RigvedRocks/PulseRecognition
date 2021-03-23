from app import app
from flask import render_template,flash,request
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,SpatialDropout2D,Dropout
import tensorflow.keras.backend as k
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
mymodel=load_model('app/static/model/my_model.h5')

IMAGE_FOLDER='app/static/upload'
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
   
   #if request.method=='POST':


       f=request.files['file']
       f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
       full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
       img=image.load_img(full_filename,target_size=(210,210),color_mode='rgb')
       img=image.img_to_array(img)
       img=np.expand_dims(img,axis=0)
       num=int(mymodel.predict_classes(img)[0])
       dal = {0: 'Arhar',1:'Chana',2:'Masoor',3:'Moong',4:'Toor',5:'Urad'}
       prediction = dal[num]
       return render_template('predict.html',image=str(full_filename),prediction=prediction)

if __name__== "__main__":
    app.run(debug=True)

