import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, TensorBoard

def preprocess_and_train_model(csv_file):
    try:
        data = pd.read_csv(csv_file, delimiter='|')
    except FileNotFoundError:
        print("Ошибка: Файл данных не найден.")
        return None

    X = data['utterance']
    y = data['importance']

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

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    model.fit(X_train_pad, y_train, epochs=5, validation_split=0.2, callbacks=[checkpoint, tensorboard])

    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f'Accuracy: {accuracy}')

    model.save('priority_classification_model.keras')
    return model, tokenizer, label_encoder

def evaluate_model(model, tokenizer, label_encoder, new_text):
    new_text_seq = tokenizer.texts_to_sequences([new_text])
    new_text_pad = pad_sequences(new_text_seq, maxlen=model.layers[0].input_shape[1])
    prediction = model.predict(new_text_pad)
    predicted_label = label_encoder.classes_[prediction.argmax(axis=-1)[0]]
    print(f"Predicted Importance: {predicted_label}")

if __name__ == "__main__":
    csv_file = 'train_data.csv'
    model, tokenizer, label_encoder = preprocess_and_train_model(csv_file)

    new_text = "I bought a product, could I modify my purchase?"
    evaluate_model(model, tokenizer, label_encoder, new_text)
