from predictfunctions import *
import cv2

model, classes, colors, output_layers = load_yolo()

img=cv2.imread('b1.jpeg')
        
img,class_ids=getpredict(img)

classname=[classes[class_ids[i]] for i in range(len(class_ids))]

print(classname)
