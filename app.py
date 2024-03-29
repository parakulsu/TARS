from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

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

data = pd.read_csv('dataset.csv')

X = data['key'] 
y = data['value'] 

le = LabelEncoder()
y = le.fit_transform(y)

max_words = 5000
max_len = 50


tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X = pad_sequences(sequences, maxlen=max_len)


loaded_model = keras.models.load_model('RNN_model.h5')

@app.route("/Sentiment", methods=["GET", "POST"])
def sentiment():
    sentiment_result = None

    if request.method == "POST":

        user_input = request.form["user_input"]
        new_data = [user_input]
        sequences = tokenizer.texts_to_sequences(new_data)
        new_data_transformed = pad_sequences(sequences, maxlen=max_len)
        predicted_result = loaded_model.predict(new_data_transformed)

        print("Predicted probabilities:", predicted_result)
        predicted_class = tf.argmax(predicted_result, axis=1).numpy()[0]

        if predicted_class == 0:
            icon = '/static/images/confused.png'
    
        elif predicted_class == 1:
            icon = '/static/images/smile.png'

        else:
             icon = '/static/images/sad.png'

           
        sentiment_result = {"user_input": user_input, "icon": icon}

        return jsonify({"status": "SUCCESS", "data": {"sentiment_result": sentiment_result}}) 

if __name__ == '__main__':
    app.run(host='0.0.0.0')
