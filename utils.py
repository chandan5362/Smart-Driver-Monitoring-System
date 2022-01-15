import cv2
import numpy as np
import keras
from math import cos, sin, pi
import efficientnet.keras as efn
import dlib
detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

class WHENet:
    def __init__(self, snapshot=None):
        base_model = efn.EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
        out = base_model.output
        out = keras.layers.GlobalAveragePooling2D()(out)
        fc_yaw = keras.layers.Dense(name='yaw_new', units=120)(out) # 3 * 120 = 360 degrees in yaw
        fc_pitch = keras.layers.Dense(name='pitch_new', units=66)(out)
        fc_roll = keras.layers.Dense(name='roll_new', units=66)(out)
        self.model = keras.models.Model(inputs=base_model.input, outputs=[fc_yaw, fc_pitch, fc_roll])
        if snapshot!=None:
            self.model.load_weights(snapshot)
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = np.array(self.idx_tensor, dtype=np.float32)
        self.idx_tensor_yaw = [idx for idx in range(120)]
        self.idx_tensor_yaw = np.array(self.idx_tensor_yaw, dtype=np.float32)

    def get_angle(self, img):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = img/255
        img = (img - mean) / std
        predictions = self.model.predict(img, batch_size=8)
        yaw_predicted = softmax(predictions[0])
        pitch_predicted = softmax(predictions[1])
        roll_predicted = softmax(predictions[2])
        yaw_predicted = np.sum(yaw_predicted*self.idx_tensor_yaw, axis=1)*3-180
        pitch_predicted = np.sum(pitch_predicted * self.idx_tensor, axis=1) * 3 - 99
        roll_predicted = np.sum(roll_predicted * self.idx_tensor, axis=1) * 3 - 99
        return yaw_predicted, pitch_predicted, roll_predicted


def softmax(x):
    x -= np.max(x,axis=1, keepdims=True)
    a = np.exp(x)
    b = np.sum(np.exp(x), axis=1, keepdims=True)
    return a/b


def rescaleFrame(frame, scale=0.50):
    #for images, video and live video
    width = int(frame.shape[1]*scale)
    # width = 240
    height = int(frame.shape[0]*scale)
    # height = 320

    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def GET_FRAME_COUNT(path):
    cap = cv2.VideoCapture(path)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    length = int(cv2.VideoCapture.get(cap, property_id))
    return length


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img


def CroppedFrame(face):
    frames = []
    rects = detector(face)
    rect_arr = [r.rect for r in rects]
    return rect_arr
