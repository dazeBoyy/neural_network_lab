import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split



class Perceptron:
    def __init__(self, input_size, num_classes, lr=0.001):
        self.W = np.random.rand(input_size, num_classes) - 0.5
        self.b = np.random.rand(num_classes) - 0.5
        self.lr = lr

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.W) + self.b
        return self.activation(linear_output)

    def train(self, X_train, y_train, epochs=500):
        for epoch in range(epochs):
            for X, y in zip(X_train, y_train):
                y_pred = self.predict(X)
                error = y - y_pred
                self.W += self.lr * np.outer(X, error)
                self.b += self.lr * error
            print(f"Epoch {epoch+1}/{epochs} complete")

    def test(self, X_test, y_test):
        correct = 0
        for X, y in zip(X_test, y_test):
            y_pred = self.predict(X)
            if np.array_equal(y, y_pred):
                correct += 1
        accuracy = correct / len(y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".PNG") or filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path)
                img = img.convert('L')
                img = img.resize((28, 28))
                img = np.array(img).flatten()
                label = filename[0]
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Ошибка при обработке файла {img_path}: {e}")
    if len(images) == 0:
        raise ValueError("Не найдено изображений для загрузки")
    return np.array(images), np.array(labels)

# Путь к папке с изображениями
folder_path = r'C:\Users\Павел\PycharmProjects\photorecognaize\photo'

# Загрузка изображений
X, y = load_images_from_folder(folder_path)

# Нормализация данных
X = X / 255.0

# Преобразование меток в бинарный формат
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42)

# Инициализация и обучение персептрона
input_size = X_train.shape[1]
num_classes = y_train.shape[1]

p = Perceptron(input_size, num_classes, lr=0.001)
p.train(X_train, y_train, epochs=500)
p.test(X_test, y_test)

def predict_image(image_path, perceptron, label_binarizer):
    try:
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((28, 28))
        img = np.array(img).flatten() / 255.0
        prediction = perceptron.predict([img])
        return label_binarizer.inverse_transform(prediction)[0]
    except Exception as e:
        print(f"Ошибка при предсказании изображения {image_path}: {e}")
        return None

# Тестирование на отдельном изображении
image_path = r'C:\Users\Павел\PycharmProjects\photorecognaize\photo_test\D.PNG'

if os.path.exists(image_path) and os.access(image_path, os.R_OK):
    predicted_label = predict_image(image_path, p, lb)
    print(f"The predicted label is: {predicted_label}")
else:
    print(f"Проблемы с доступом к файлу {image_path}")