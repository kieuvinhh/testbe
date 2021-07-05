from flask import Flask
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from deepface import DeepFace

app = Flask(__name__)


model = load_model('MobileNetV3Large.h5')


def ConvBase64toImage(img_base64):
  try:
    image = np.fromstring(base64.b64decode(img_base64), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
    return image
  except:
    return None


def preprocess_image(data):
    image = Image.fromarray(data, 'RGB')
    image = image.resize((224,224))
    image = np.array(image)
    image = np.expand_dims(image, axis = 0)
    return image


def detect_face(img_path):
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
    detected_face = DeepFace.detectFace( img_path , detector_backend=backends[2])
    return detected_face


def predict_age(image):
    input = preprocess_image(image)
    result = model.predict(input)
    return int(result)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/image', methods=['POST'])
@cross_origin(origin='*')
def process():
    img_arg_base64 = request.form.get('img')
    img1 = ConvBase64toImage(img_arg_base64)
    img2 = detect_face(img1)
    age = predict_age(img2)
    return str(age)

if __name__ == '__main__':
  app.run()