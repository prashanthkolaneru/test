import argparse
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import base64
import cv2
import numpy as np
import uuid
import time
import os

config_path='data/yolov3.cfg'
weights_path='data/yolov3_31000.weights'
names_path='data/obj.names'

UPLOAD_FOLDER = 'uploads'
TEST_FOLDER = 'test'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','png'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def mai(image_path):
    print("starting program . . .")
    CONF_THRESH, NMS_THRESH = 0.5, 0.5
 
    # Load the network
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get the output layer from YOLO
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

    # Draw the filtered bounding boxes with their class to the image
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for index in indices:
        x, y, w, h = b_boxes[index]
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
        cv2.putText(img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index], 2)

    #cv2.imshow("image", img)
    cv2.imwrite('test/image.jpg', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEST_FOLDER'] = TEST_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', imagesource='../uploads/template.jpg', imagesource2='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            outputfile='image.jpg'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            op_path = os.path.join(app.config['TEST_FOLDER'], outputfile)
            
            
            file.save(file_path)
            mai(file_path)
            
            print(file_path)
            
            print(op_path)
            
            filename = my_random_string(6) + filename
            outputfile = my_random_string(6) + outputfile
            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.rename(op_path, os.path.join(app.config['TEST_FOLDER'], outputfile))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html',imagesource='../uploads/'+filename, imagesource2='../test/'+outputfile)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
@app.route('/test/<filename>')
def output_file(filename):
    return send_from_directory(app.config['TEST_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=5000)
