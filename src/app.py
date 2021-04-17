from flask import Flask,render_template,Response,request,redirect,jsonify
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from psycopg2 import sql
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import datetime
from gaze_tracking import GazeTracking
from math import sqrt
import imutils
gaze = GazeTracking()
from imutils.video import VideoStream
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import mysql.connector
os.environ['TF_XLA_FLAGS']= '--tf_xla_enable_xla_devices'
OPENCV_PYTHON_DEBUG=1
app = Flask(__name__)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
mydb = mysql.connector.connect(
    host="localhost",
    database="facial_emotion", #name of your database
    user="root",
    password="",
)
cursor = mydb.cursor(buffered=True)
global timestamp 
timestamp=datetime.datetime.now()

@app.route('/send',methods=["GET","POST"])
def send():
    global table1
    table1=request.form['user_name'] 
    return render_template('home.html')

camera = cv2.VideoCapture(0)  

def gen_frames():  # generate frame by frame from camera
    global emotion_dict
    global maxindex
    query = """SELECT count(*) FROM information_schema.TABLES WHERE (TABLE_SCHEMA ='facial_emotion') AND (TABLE_NAME='%s')""" %(table1)
    cursor.execute(query)
    mydb.commit()
    
    if cursor.fetchone()[0]==1:
        print("table exists")
    else:
        query1="""CREATE TABLE %s(id INT AUTO_INCREMENT PRIMARY KEY,data VARCHAR(255),time timestamp,emotionid int)""" %(table1)
        cursor.execute(query1)
        mydb.commit()

        
                
   
    
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:
            model.load_weights('model.h5')
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
            gaze.refresh(frame)

            frame = gaze.annotated_frame()
            text = ""

            if gaze.is_blinking():
                text = "Blinking"
            elif gaze.is_right():
                text = "Looking right"
            elif gaze.is_left():
                text = "Looking left"
            elif gaze.is_center():
                text = "Looking center"

            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            
            facecasc =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
            try:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    global buffer
                    ret,buffer = cv2.imencode('.jpg',frame)
                    frame = buffer.tobytes()
                         

                    global s
                    s = "Emotion:" + emotion_dict[maxindex]

                    query2 = """INSERT INTO {} (data,time,emotionid) VALUES ('%s','%s','%d')""".format(table1) % (s,timestamp,maxindex)  #Database query
                    cursor.execute(query2)
                    mydb.commit()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(buffer) + b'\r\n')  # concat frame one by one and show result
            except:
                print("error occured")

@app.route('/getAnalytics')

def getAnalytics():
    try:
        table2=request.args['name']
        query2 = """SELECT COUNT(emotionid) as Count ,time ,data FROM {} GROUP BY emotionid""".format(table2)  #Database query
    # query2= """SELECT COUNT(emotionid) as COUNT ,time ,data FROM  ${table2} GROUP BY emotionid""" 
        cursor.execute(query2)
        #rows=cursor.fetchone()
        columns = cursor.description
        result = []
        for value in cursor.fetchall():
            tmp = {}
            for (index,column) in enumerate(value):
                tmp[columns[index][0]] = column
            result.append(tmp)
        response = jsonify(result)

        return response
    except Exception as e:
        return e
            


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
        
    """Video streaming home page."""
    return render_template('index.html')

@app.route("/stream")
def stream():
    def generate():
        for s in range(500):
            yield "{}\n".format(sqrt(s))
    return app.response_class(generate(), mimetype="text/plain")

if __name__ == '__main__':
    app.run(debug=True)