import cv2
import time
from utils import *
from iris import iris_tracking
from object_detection import webcam_detect
import mediapipe as mp

from mediapipe_headpose import headpose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def get_coordinates(landmarks, index, image):
    """
    Get coordinates of a landmark.

    """
    h, w, c = image.shape
    coordinate = int(landmarks[index].x*w), int(landmarks[index].y*h)
    return coordinate


def eucledian(x, y):
    """
    Calculate eucledian distance.

    """
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5



#-------------------------------------------------------------------------
# For webcam input:
model = WHENet('models/WHENet.h5')
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False

    #iris tracking
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    beg = time.time()
    try:
        image = headpose(image, model)
    except:
        continue
    
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        #eye landmarks
        #for right eye
        pr1, pr2, pr3, pr4, pr5, pr6 = get_coordinates(landmarks, 33, image), get_coordinates(landmarks, 160, image), get_coordinates(landmarks, 158, image), get_coordinates(landmarks, 133, image), get_coordinates(landmarks, 153, image), get_coordinates(landmarks, 144, image)
        EAR_right = (eucledian(pr2,pr6)+eucledian(pr3,pr5))/(2*eucledian(pr1,pr4))
        #for left eye
        pl1, pl2, pl3, pl4, pl5, pl6 = get_coordinates(landmarks, 362, image),get_coordinates(landmarks, 385, image), get_coordinates(landmarks, 387, image), get_coordinates(landmarks, 263, image), get_coordinates(landmarks, 373, image), get_coordinates(landmarks, 380, image)
        EAR_left = (eucledian(pl2, pl6)+eucledian(pl3,pl5))/(2*eucledian(pl1,pl4))

        cv2.putText(image, "Left EAR: :{:.2f}".format(EAR_left), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
        cv2.putText(image, "Right EAR: {:.2f}".format(EAR_right), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
        # print("EAR_right: ", EAR_right)
        # print("EAR_left: ", EAR_left)
    
        #mouth landmarks
        ml, mlu, mld, mru, mrd, mr = get_coordinates(landmarks, 78, image), get_coordinates(landmarks, 82, image), get_coordinates(landmarks, 87, image), get_coordinates(landmarks, 312, image), get_coordinates(landmarks, 317, image), get_coordinates(landmarks, 308, image)
        MAR_mouth = (eucledian(mlu,mld)+eucledian(mru,mrd))/(2*eucledian(ml,mr))
        # print("EAR_mouth: ", MAR_mouth)
        cv2.putText(image, "MAR_mouth: {:.2f}".format(MAR_mouth), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 255), 1)

        #drawing the useful landmarks
        indices = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 78, 82, 87, 312, 317, 308]
        for i in indices:
            image = cv2.circle(image, get_coordinates(landmarks, i, image), 2, (0, 0, 255), -1)
        
        #display warning if EAR is too low
        if EAR_right < 0.2 or EAR_left < 0.2:
            cv2.putText(image, "WARNING: EAR too low!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        if MAR_mouth > 0.2:
            cv2.putText(image, "WARNING: Mouth open!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

         
    image = webcam_detect(image)
    end = time.time()
    # print("FPS: ", 1/(end-beg))
    # Flip the image horizontally for a selfie-view display.
    
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()