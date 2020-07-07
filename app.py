import flask
from flask import render_template
from predictfunctions import *
import cv2

app=flask.Flask(__name__)

@app.route("/")
def home():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predict():
        
            print('post')
            img=cv2.imread('b1.jpeg')
        
            img,class_ids=getpredict(img)
            model, classes, colors, output_layers = load_yolo()
            classname=[classes[class_ids[i]] for i in range(len(class_ids))]
            print(classname)
            return 'sucessful'

if __name__=="__main__":
   app.run(debug=True)
