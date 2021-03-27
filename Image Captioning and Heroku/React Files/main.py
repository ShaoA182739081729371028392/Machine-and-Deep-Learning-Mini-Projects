from flask import Flask, request, jsonify
from captioner import load_captioner, process_image
import PIL.Image as Image
import json
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import io
import os
# Load in PyTorch Model and State Dictionary
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = load_captioner() 
@app.route("/process", methods = ['POST'])
@cross_origin()
def process():
    '''
    Runs inference on the Image Given by the React Server. Loads the image in, then returns the caption.
    '''
    files = request.files['file']
    vals = process_image(model, Image.open(io.BytesIO(files.read())))
    # Process Model
    return jsonify(vals)
if __name__ == '__main__':
    app.run(host = 'localhost', port = '5000', debug = True)