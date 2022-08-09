import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from vidgear.gears import CamGear
import sys
import os
import uuid

class CarDetection:
    def __init__(self,url):
        self.URL = url
        self.model = self.load_model()
        self.line = (490, 525),(675,580)
        self.counter = 0

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        return model

    def get_stream_from_url(self):
        options = {"STREAM_RESOLUTION": "720p"}
        stream = CamGear(source=self.URL, stream_mode = True, logging=True, **options).start()
        return stream

    def detect(self, image):
        car = []
        pos = []
        results = self.model(image)
        labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
        for i in range(len(labels)):
            if labels[i] == 2:
                car.append(labels[i])
                pos.append(cord[i])
        return car, pos

    def center(self, x1,y1,x2,y2):
        x = (x1+x2)/2
        y = (y1+y2)/2
        return x,y
        
    def check_line(self,x,y):
        xLine, yLine = self.line
        if x> xLine[0] and x < xLine[1]:
            if y < yLine[0] and y > yLine[1]:
                return True
            
        return False

    def draw(self,results, image):
        cars, cord = results
        width, height = image.shape[1], image.shape[0]
        for i in range(len(cars)):
            row = cord[i]
            color = (0, 255, 0)
            x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, f"Total Cars crossed: {self.counter}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            x,y = self.center(x1,y1,x2,y2)
            state = self.check_line(x,y)
            cv2.line(image,self.line[0],self.line[1], color,4)

            if state:
                self.counter+=1
                self.save(image)

        return image

    def save(self, image):
        IMAGES_PATH = os.path.join('./SaveImages')
        cv2.imwrite(os.path.join(IMAGES_PATH,'{}.jpg'.format(str(uuid.uuid1()))), image)

    def __call__(self):   
        stream = self.get_stream_from_url()
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            frame = stream.read()

            if frame is None:
                break

            results = self.detect(frame)
            img = self.draw(results,frame)   

            cv2.imshow('YOLO', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

link = sys.argv[1]
a = CarDetection(link)
a()