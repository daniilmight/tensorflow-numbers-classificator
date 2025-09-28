import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# ============================================================
# Пути и директории проекта
# ============================================================
# BASE_DIR - основная директория проекта, в которой будут храниться все файлы
# EXPORT_DIR - директория для сохранения frozen graph и тестовых данных
# CHECKPOINT_DIR - директория для сохранения чекпоинтов модели (лучшие веса)
BASE_DIR = r"D:\Codding\tensorflow-numbers-classificator\tensorflow-numbers-classificator"
EXPORT_DIR = os.path.join(BASE_DIR, "export")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Загрузка и предобработка данных MNIST
# ============================================================
# Загружаем стандартный датасет MNIST (изображения 28x28 пикселей)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Преобразуем данные:
# - добавляем размерность канала (чёрно-белые изображения → (28,28,1))
# - переводим в float32
# - нормализуем в диапазон [0,1]
x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

# Ограничиваем размер датасета для ускорения обучения
TRAIN_LIMIT = 60000 # - количество картинок для обучения
TEST_LIMIT = 10000 # - количество картинок для теста

x_train, y_train = x_train[:TRAIN_LIMIT], y_train[:TRAIN_LIMIT]
x_test, y_test = x_test[:TEST_LIMIT], y_test[:TEST_LIMIT]

print(f"Используем {len(x_train)} обучающих и {len(x_test)} тестовых изображений")

# ============================================================
# Архитектура сверточной нейросети (CNN)
# ============================================================
# Модель принимает вход (28x28x1) и выполняет классификацию на 10 классов
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1), name="input_image"),

    # Первый сверточный слой:
    # filters = 32 (количество фильтров), kernel_size=3 (размер фильтра 3x3),
    # activation="relu" (функция активации ReLU),
    # padding="same" (выход имеет ту же размерность, что и вход)
    keras.layers.Conv2D(32, 3, activation="relu", padding="same", name="conv_1"),
    keras.layers.MaxPooling2D(pool_size=2, name="pool_1"),  # уменьшаем размерность в 2 раза

    # Второй сверточный слой:
    # filters = 64 (увеличиваем количество фильтров для извлечения большего числа признаков)
    keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="conv_2"),
    keras.layers.MaxPooling2D(pool_size=2, name="pool_2"),

    # Преобразование карт признаков в одномерный вектор
    keras.layers.Flatten(name="flatten"),

    # Полносвязный скрытый слой: 256 нейронов, активация ReLU
    keras.layers.Dense(256, activation="relu", name="dense1_hidden"),

    # Полносвязный скрытый слой: 32 нейронов, активация ReLU
    keras.layers.Dense(32, activation="relu", name="dense2_hidden"),

    # Выходной слой: 10 нейронов (по числу классов MNIST), softmax для вероятностей
    keras.layers.Dense(10, activation="softmax", name="predictions")
])

# ============================================================
# Компиляция модели
# ============================================================
# optimizer - алгоритм оптимизации (Adam, скорость обучения 0.001)
# loss - функция ошибки (sparse_categorical_crossentropy подходит для целых меток классов)
# metrics - метрика качества (accuracy - доля правильных ответов)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# Настройки обучения
# ============================================================
EPOCHS = 64       # количество эпох обучения (полных проходов по датасету)
BATCH_SIZE = 32  # размер батча (количество изображений в одной итерации обучения)

# Callbacks - вспомогательные функции для обучения
callbacks = [
    # Сохраняем только лучшую модель (по метрике val_accuracy)
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    ),
    # Останавливаем обучение, если валидация не улучшается 5 эпох подряд
    keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),
    # Логи для TensorBoard (можно смотреть графики обучения в реальном времени)
    keras.callbacks.TensorBoard(
        log_dir=os.path.join(BASE_DIR, "logs"),
        histogram_freq=1
    )
]

# Запуск обучения
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,  # 10% данных выделяются на валидацию
    callbacks=callbacks,
    verbose=2
)

# ============================================================
# Сохранение замороженного графа (Frozen Graph)
# ============================================================
# Преобразуем модель в статический граф с фиксированными весами
full_model = tf.function(lambda x: model(x))
concrete_func = full_model.get_concrete_function(
    tf.TensorSpec([1, 28, 28, 1], tf.float32)
)

# Замораживаем граф (переводим переменные в константы)
frozen_func = convert_variables_to_constants_v2(concrete_func)
frozen_graph_def = frozen_func.graph.as_graph_def()

# Сохраняем frozen graph в файл
tf.io.write_graph(frozen_graph_def, EXPORT_DIR, "mnist_frozen_graph.pb", as_text=False)
print(f"Frozen graph сохранен в {os.path.join(EXPORT_DIR, 'mnist_frozen_graph.pb')}")

# ============================================================
# Сохранение тестового примера
# ============================================================
# Сохраняем одно изображение и метку к нему (для последующего инференса)
sample = x_test[0:1]  # batch=1
sample.astype("float32").tofile(os.path.join(EXPORT_DIR, "sample_input.bin"))
with open(os.path.join(EXPORT_DIR, "sample_label.txt"), "w") as f:
    f.write(str(int(y_test[0])))

print("Файлы sample_input.bin и sample_label.txt сохранены")
