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


data = pd.read_csv('dataset.csv')

# กำหนด X และ y จากข้อมูล
X = data['key'] # รวมข้อมูลใน 'key' และ 'value'
y = data['value']  # แปลงค่าใน 'h' เป็นข้อความ

# ทำการแปลง Label เป็นตัวเลข
le = LabelEncoder()
y = le.fit_transform(y)

# Tokenization และ Padding ข้อมูล
max_words = 5000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X = pad_sequences(sequences, maxlen=max_len)

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้างโมเดล RNN
model = Sequential()
model.add(Embedding(max_words, 50, input_length=max_len))
model.add(SimpleRNN(100))
model.add(Dense(3, activation='softmax'))

# คอมไพล์และฝึกโมเดล
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# ประเมินประสิทธิภาพของโมเดล
accuracy = model.evaluate(X_test, y_test)[1]
# print(model.evaluate(X_test, y_test))
print(f'Accuracy: {accuracy}')

# บันทึกโมเดล
model.save('RNN_model.h5')