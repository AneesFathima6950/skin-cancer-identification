# Importing essential python libraries

from __future__ import division, print_function
import os
import numpy as np
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
# Flask utils
from flask import Flask,  url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from heatmap import save_and_display_gradcam,make_gradcam_heatmap



os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Define a flask app
app = Flask(__name__, static_url_path='')


app.config['HEATMAP_FOLDER'] = 'heatmap'
app.config['UPLOAD_FOLDER'] = 'uploads'
# Model saved with Keras models.save()
MODEL_PATH = 'models/keras_model.h5'
#model = load_model('./model.h5', compile = False)

#Load your trained models
model = load_model(MODEL_PATH)
        # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')


class_dict = {
              0 : "Vascular lesion",
              1 : "Pigmented benign keratosis",
              2: "Melanoma",
              3 : "Basal cell carcinoma",
              4 : "Actinic keratosis" }

@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



def model_predict(img_path, model):
    
    img = Image.open(img_path).resize((224,224)) #target_size must agree with what the trained models expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
   
    preds = model.predict(img)[0]
    prediction = sorted(
      [(class_dict[i], round(j*100, 2)) for i, j in enumerate(preds)],
      reverse=True,
      key=lambda x: x[1]
  )
    print(prediction)
    return prediction,img


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        file_name=os.path.basename(file_path)
        # Make prediction
        pred,img = model_predict(file_path, model)

    return render_template('predict.html',file_name=file_name,result=pred)

if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)

