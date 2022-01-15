import cv2
import time
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
from iris import iris_tracking
from utils import *


def process_detection( model, img, box):
  h, w, c = img.shape
  x_min, y_min = int(w * box.xmin), int(h * box.ymin)
  x_max = x_min + int(w * box.width)
  y_max = y_min + int(h * box.height)


  img_rgb = img[y_min:y_max,x_min:x_max,:]
  img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
  img_rgb = cv2.resize(img_rgb, (224, 224))
  img_rgb = np.expand_dims(img_rgb, axis=0)

  cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), 2)
  yaw, pitch, roll = model.get_angle(img_rgb)
  yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
  if yaw <-25:
    cv2.putText(img, "WARNING: Looking Left!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
  elif yaw > 25:
    cv2.putText(img, "WARNING: Looking Right!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
  
  if pitch <-25:
    cv2.putText(img, "WARNING: Looking Down!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
  elif pitch > 25:
    cv2.putText(img, "WARNING: Looking Up!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
 


  #display yaw, pitch and roll
  cv2.putText(img, "YAW: {:.2f}".format(yaw), (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
  cv2.putText(img, "PITCH: {:.2f}".format(pitch), (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
  cv2.putText(img, "ROLL: {:.2f}".format(roll), (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

  draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min)//2 )

  #display yaw, roll and pitch

  cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
  cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
  cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
  return img


def headpose(image, model):

  with mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5) as face_detection:

    results = face_detection.process(image)
    
    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    if results.detections:
      for detection in results.detections:
        
        # print(detection)
        box = detection.location_data.relative_bounding_box
        image = iris_tracking(image, box)
        image = process_detection(model, image, box)
    return image
