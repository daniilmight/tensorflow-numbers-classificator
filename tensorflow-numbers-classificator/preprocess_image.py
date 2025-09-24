import numpy as np  # для работы с массивами
from PIL import Image  # для загрузки и конвертации изображений
import sys  # для аргументов командной строки

if len(sys.argv) < 3: sys.exit("Usage: python preprocess_image.py <input_image> <output_bin>")  # проверка аргументов

img = Image.open(sys.argv[1]).convert('L')  # открываем изображение и конвертируем в grayscale
img = img.resize((28,28))  # изменяем размер на 28x28 пикселей
arr = np.array(img).astype(np.float32)/255.0  # конвертируем в float32 и нормализуем
arr = np.expand_dims(arr, -1)  # добавляем канал (28,28) -> (28,28,1)
arr.tofile(sys.argv[2])  # сохраняем как binary файл
