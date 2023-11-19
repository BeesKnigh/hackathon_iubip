import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import (Embedding, LSTM, Dense)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint, TensorBoard

def preprocess_and_train_model(csv_file, employee_competencies_file):
    try:
        data = pd.read_csv(csv_file, delimiter='|')
        employee_competencies = pd.read_csv(employee_competencies_file, delimiter='|')
    except FileNotFoundError:
        print("Ошибка: Один или оба файла данных не найдены.")
        return None

    X = data['utterance']
    y = data['request']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq)
    X_test_pad = pad_sequences(X_test_seq, maxlen=X_train_pad.shape[1])

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=X_train_pad.shape[1]))
    model.add(LSTM(100))
    model.add(Dense(len(set(y_train)), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    model.fit(X_train_pad, y_train, epochs=5, validation_split=0.2, callbacks=[checkpoint])

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=X_train_pad.shape[1]))
    model.add(LSTM(100))
    model.add(Dense(len(set(y_train)), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    model.fit(X_train_pad, y_train, epochs=5, validation_split=0.2,
              callbacks=[checkpoint, tensorboard])  # Добавлен TensorBoard

    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f'Accuracy: {accuracy}')

    model.save('customer_routing_model.keras')
    return model, tokenizer, label_encoder, employee_competencies

def find_employee_category(employee_competencies, predicted_label):
    column_name = f'is_competent_in_{predicted_label}'
    if column_name in employee_competencies.columns:
        employee_row = employee_competencies.loc[employee_competencies[column_name] == 1]
        if not employee_row.empty:
            return employee_row['employee'].values[0]
        else:
            print(f"Ошибка: Сотрудник для категории '{predicted_label}' не найден.")
            return None
    else:
        print(f"Ошибка: Категория '{predicted_label}' не найдена в employee_competencies.")
        return None



# обработка того примера:
def evaluate_model(model, tokenizer, label_encoder, employee_competencies, new_text):
    new_text_seq = tokenizer.texts_to_sequences([new_text])
    new_text_pad = pad_sequences(new_text_seq, maxlen=model.layers[0].input_shape[1])
    prediction = model.predict(new_text_pad)
    predicted_label = label_encoder.classes_[prediction.argmax(axis=-1)[0]]
    print(f"Predicted Label: {predicted_label}")

    employee = find_employee_category(employee_competencies, predicted_label)
    if employee:
        print(f"Assigned Employee: {employee}")

if __name__ == "__main__":
    csv_file = 'train_data.csv'
    employee_competencies_file = 'employee_competencies.csv'

    model, tokenizer, label_encoder, employee_competencies = preprocess_and_train_model(csv_file, employee_competencies_file)

    # Пример для проверки и всяких тестиков:
    new_text = "i bought a product, could I modify my fucking purchase?"
    evaluate_model(model, tokenizer, label_encoder, employee_competencies, new_text)
