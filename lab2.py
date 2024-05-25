import numpy as np
import os
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование данных
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Преобразование меток в бинарный формат
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Построение модели
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=2, batch_size=128, validation_split=0.2)

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')


def load_custom_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".PNG") or filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path)
                img = img.convert('L')
                img = img.resize((28, 28))
                img = np.array(img).astype('float32') / 255
                img = np.expand_dims(img, axis=-1)
                images.append(img)
                labels.append(filename[0])
            except Exception as e:
                print(f"Ошибка при обработке файла {img_path}: {e}")
    if len(images) == 0:
        raise ValueError("Не найдено изображений для загрузки")
    return np.array(images), labels

def predict_custom_image(image_path, model):
    try:
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((28, 28))
        img = np.array(img).astype('float32') / 255
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)
        return predicted_label
    except Exception as e:
        print(f"Ошибка при предсказании изображения {image_path}: {e}")
        return None


# Путь к папке с вашими изображениями
image_file_path = r'C:\Users\Павел\PycharmProjects\photorecognaize\photo_test\3.PNG'

# Предсказание для одного изображения
pred = predict_custom_image(image_file_path, model)
print(f"Predicted label for {image_file_path}: {pred}")