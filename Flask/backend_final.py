import json
import tensorflow
import keras
import os
import tensorflow as tf
import cv2
from keras.models import load_model
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from flask import jsonify

app = Flask(__name__)

@app.route('/upload-video', methods=['POST'])
def upload_video():
    file = request.files['video']
    file.save('uploadedvideos/video.mp4')
    source_1 = 'uploadedvideos/video.mp4'
    source_2 = 'uploadedvideos/video.mp4'
    dest = 'content/Images'
    model = 'model/model_cricket.h5'
    main(source_1, source_2, dest, model)
    analysis(results)
    print(results)
    return json.dumps(results)

try:
    model = tf.keras.models.load_model('model/model_cricket.h5')
    print("Model loaded")
except Exception as e:
    print(str(e))

def seperate_images(source, dest):

    if os.path.exists(dest):
        print("Found an existing path. Removing files...")
        files = os.listdir(dest)
        for f in files:
            os.remove(os.path.join(dest, f))
    else:
      os.mkdir(dest)

    vidcap = cv2.VideoCapture(source)
    def getFrame(sec, count):
      vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
      hasFrames,image = vidcap.read()
      if hasFrames:
          file_name = "cricket_" + str(count) + ".jpg"
          cv2.imwrite(os.path.join(dest, file_name), image)
      return hasFrames
      
    sec = 0
    frameRate = 1.0
    count=1
    success = getFrame(sec, count)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec, count)
      
    total_files = os.listdir(dest)
    print("Total shots processed:", len(total_files))

def predictor(dest, model):
    files = os.listdir(dest)
    base_model = load_model(model)

    predictions = []
    for f in files:
        path = os.path.join(dest, f)

        img = load_img(path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        datagen = ImageDataGenerator()
        gen = datagen.flow(img)

        prediction = base_model.predict(gen, verbose=0)
        prediction = np.argmax(prediction, axis=1)
        predictions.append(prediction)

    return predictions

def max_performance(maximum, ana_drive, ana_legglance_flick, ana_pullshot):
    max_class = ""
    if maximum == ana_drive:
      max_class = "Drive"
    elif maximum == ana_legglance_flick:
      max_class = "Legglance Flick"
    elif maximum == ana_pullshot:
      max_class = "Pullshot"
    else:
      max_class = "Sweep"

    return max_class


def summarizer(predictions):
    drive = 0
    legglance_flick = 0
    pullshot = 0
    sweep = 0

    for prediction in predictions:
        if prediction == 0:
            drive += 1
        elif prediction == 1:
            legglance_flick += 1
        elif prediction == 2:
            pullshot += 1
        else:
            sweep += 1

    total = drive + legglance_flick + pullshot + sweep

    if drive != 0:
      ana_drive = round(((drive/total)*100), 0)
    else:
      ana_drive = 0

    if legglance_flick != 0:
      ana_legglance_flick = round(((legglance_flick/total)*100), 0)
    else:
      legglance_flick = 0

    if pullshot != 0:
      ana_pullshot = round(((pullshot/total)*100), 0)
    else:
      ana_pullshot = 0

    if sweep != 0:
      ana_sweep = round(((sweep/total)*100), 0)
    else:
      ana_sweep = 0

    maximum = max(ana_drive, ana_legglance_flick, ana_pullshot, ana_sweep)
    max_class = max_performance(maximum, ana_drive, ana_legglance_flick, ana_pullshot)

    return [ana_drive, ana_legglance_flick, ana_pullshot, ana_sweep, 
            max_class, maximum]



results = {
    "Player_1" : [],
    "Player_2" : []
}

def analysis(results):
    Drive_dif = results['Player_2'][0] - results['Player_1'][0]
    Legglance_flick_dif = results['Player_2'][1] - results['Player_1'][1]
    Pullshot_dif = results['Player_2'][2] - results['Player_1'][2]
    Sweep_dif = results['Player_2'][3] - results['Player_1'][3]

    maximum = max(Drive_dif, Legglance_flick_dif, Pullshot_dif, Sweep_dif)
    max_class = max_performance(maximum, Drive_dif, Legglance_flick_dif, Pullshot_dif)

    print("====================================================")
    print('================== Analysis Report =================')
    print("====================================================\n")
    print('Drive improve rate             : {0}%'.format(Drive_dif))
    print('Legglance Flick improve rate   : {0}%'.format(Legglance_flick_dif))
    print('Pullshot improve rate          : {0}%'.format(Pullshot_dif))
    print('Sweep improve rate             : {0}%\n'.format(Sweep_dif))
    print('Player is focused at           : {0} at {1}%'.format(max_class, maximum))
    print("====================================================")



def main(source_1, source_2, dest, model):
    sources = [source_1, source_2]
    for i, source in enumerate(sources):
        print("Processing video", source)
        seperate_images(source, dest)

        print("Analysing video")
        predictions = predictor(dest, model)

        print("Generating results")
        ana_drive, ana_legglance_flick, ana_pullshot, ana_sweep, max_class, maximum = summarizer(predictions)

        if i == 0:
          results['Player_1'] = [ana_drive, ana_legglance_flick, ana_pullshot, ana_sweep]
        else:
          results['Player_2'] = [ana_drive, ana_legglance_flick, ana_pullshot, ana_sweep]

        print("====================================================")
        print('=============== Performance Report {0} ==============='.format(i))
        print("====================================================\n")
        print('Drive                : {0}%'.format(ana_drive))
        print('Legglance Flick      : {0}%'.format(ana_legglance_flick))
        print('Pullshot             : {0}%'.format(ana_pullshot))
        print('Sweep                : {0}%\n'.format(ana_sweep))
        print('Player is focused at : {0} at {1}%'.format(max_class, maximum))
        print("====================================================")

if __name__ == '__main__':
    app.run(port=8082)  # Change the port number to 8080


  