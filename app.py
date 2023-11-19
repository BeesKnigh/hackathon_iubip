from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from customer_routing_app import preprocess_and_train_model, find_employee_category
import pandas as pd

app = Flask(__name__)

model, tokenizer, label_encoder, employee_competencies = None, None, None, None

@app.before_request
def before_request():
    global model, tokenizer, label_encoder, employee_competencies
    if model is None:
        model, tokenizer, label_encoder, employee_competencies = preprocess_and_train_model('train_data.csv', 'employee_competencies.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        new_text = request.form['user_query']
        new_text_seq = tokenizer.texts_to_sequences([new_text])
        new_text_pad = pad_sequences(new_text_seq, maxlen=model.layers[0].input_shape[1])
        prediction = model.predict(new_text_pad)
        predicted_label = label_encoder.classes_[prediction.argmax(axis=-1)[0]]
        employee = find_employee_category(employee_competencies, predicted_label)
        return render_template('result.html', result=predicted_label, employee=employee)

if __name__ == '__main__':
    app.run(debug=True)
