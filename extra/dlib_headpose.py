import cv2
import matplotlib.pyplot as plt
import numpy
from  utils import *
import pickle
import time

def process_detection( model, img, box):
    print(img.shape)
    tx, ty,bx, by = int(box.left()), int(box.top()), int(box.right()), int(box.bottom())
  

    img_rgb = img[ty:by,tx:bx,:]
    # img_rgb = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    x_min = tx
    y_min = by
    x_max = bx
    y_max = ty
    print(img_rgb.shape)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = np.expand_dims(img_rgb, axis=0)

    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), 2)
    yaw, pitch, roll = model.get_angle(img_rgb)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
    draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min)//2 )

    # if args.display == 'full':
    cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    return img



def headpose(path):
    model = WHENet('models/WHENet.h5')
    cap = cv2.VideoCapture(path)
    while True:
        try:
            ret, frame = cap.read()
            
            frame = rescaleFrame(frame) #(H, w, c)
        except:
            break
        bg= time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_rgb = frame_rgb.astype(numpy.float32)
        bbox = CroppedFrame(frame_rgb) 
        for box in bbox:
            frame = process_detection(model, frame, box)
        end = time.time()
        print("FPS: ", 1/(end-bg))
        cv2.imshow('output',frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    headpose("/Users/chandanroy/Desktop/Freelance/video1.mp4")
    