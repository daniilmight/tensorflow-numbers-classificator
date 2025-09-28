import tkinter as tk
import numpy as np
import cv2
from scipy import ndimage  # для сдвига по центру масс
from PIL import Image
import struct

# Размер холста для рисования
CANVAS_SIZE = 280  # увеличенный холст для удобства рисования
IMG_SIZE = 28      # размер выходного изображения (как в MNIST)

class DigitDrawer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw a digit")
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        # Кнопки управления
        save_btn = tk.Button(self.root, text="Save", command=self.save_digit)
        save_btn.pack()
        clear_btn = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        clear_btn.pack()

        # Массив для хранения нарисованного изображения
        self.image = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    def draw(self, event):
        # Рисуем белую кисть (цифра)
        x, y = event.x, event.y
        r = 8  # радиус кисти
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        cv2.circle(self.image, (x, y), r, 255, -1)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image.fill(0)

    def save_digit(self):
        # --- 1. Бинаризация ---
        _, thresh = cv2.threshold(self.image, 50, 255, cv2.THRESH_BINARY)

        # --- 2. Находим bounding box цифры ---
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)

        # --- 3. Вырезаем цифру ---
        digit = thresh[y:y+h, x:x+w]

        # --- 4. Масштабируем до 20x20 ---
        digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

        # --- 5. Вставляем в центр 28x28 ---
        canvas = np.zeros((28, 28), dtype=np.float32)
        start_x = (28 - 20) // 2
        start_y = (28 - 20) // 2
        canvas[start_y:start_y+20, start_x:start_x+20] = digit / 255.0

        # --- 6. Центрируем по центру масс ---
        cy, cx = ndimage.center_of_mass(canvas)
        shift_x = int(np.round(14 - cx))
        shift_y = int(np.round(14 - cy))
        canvas = ndimage.shift(canvas, shift=[shift_y, shift_x], mode='constant', cval=0.0)

        # --- 7. Сохраняем бинарный файл ---
        with open("export/sample_input.bin", "wb") as f:
            f.write(struct.pack("784f", *canvas.flatten()))

        # --- 8. Сохраняем PNG для проверки ---
        img_to_save = (canvas * 255).astype(np.uint8)
        Image.fromarray(img_to_save).save("debug_digit.png")

        print("Saved input_image.bin and debug_digit.png")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = DigitDrawer()
    app.run()
