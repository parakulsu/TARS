from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import cv2
import numpy as np

app = Flask(__name__)

target_width, target_height = 480, 480
model = keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'})

    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        h, w, _ = img.shape
        left = (w - target_width) // 2
        top = (h - target_height) // 2
        right = (w + target_width) // 2
        bottom = (h + target_height) // 2

        img = img[top:bottom, left:right]

        img = cv2.resize(img, (target_width, target_height))
        img = np.expand_dims(img, axis=0)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)

    if predicted_label == 0:
        name = 'ไม่สามารถระบุสถานที่ได้'
    elif predicted_label == 1:
        name = 'ตึกคณะวิทยาการสารสนเทศ'
    elif predicted_label == 2:
        name = 'หาดวอนนภา'
    else:
        predicted_label = -1
        name = 'ไม่สามารถระบุสถานที่ได้'

    print(predictions)
    print(predicted_label)
    return jsonify({'prediction': name})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
