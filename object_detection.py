import cv2
import numpy as np 
import argparse
import time

def load_yolo():
	net = cv2.dnn.readNet("YOLO/yolov3-tiny.weights", "YOLO/yolov3-tiny.cfg")
	classes = []
	with open("YOLO/coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()] 
	
	output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels


def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs


def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	exceptions = ['bottle', 'cell phone', 'wine glass', 'cup']
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			if label not in exceptions:
				continue
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	return img



def webcam_detect(image):
	model, classes, colors, output_layers = load_yolo()

	height, width, channels = image.shape
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

	bottle_id = classes.index('bottle')
	i=1
	if bottle_id in class_ids:
		cv2.putText(image, "WARNING: Bootle Detected!", (width-600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		i+=1
	
	#mobile phone detection
	phone_id = classes.index('cell phone')
	if phone_id in class_ids:
		cv2.putText(image, "WARNING: Mobile Phone Detected!", (width-600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		i+=1
	
	#wine glass detection
	wine_id = classes.index('wine glass')
	if wine_id in class_ids:
		cv2.putText(image, "WARNING: Wine Glass Detected!", (width-600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		i+=1

	#cup detection
	cup_id = classes.index('cup')
	if cup_id in class_ids:
		cv2.putText(image, "WARNING: Cup Detected!", (width-600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		i+=1	
	# print(class_ids)
	if len(class_ids) > 0:
		image = draw_labels(boxes, confs, colors, class_ids, classes, image)
		
	return image
