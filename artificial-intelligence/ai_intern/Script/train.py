import os
import numpy as np
from util import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
from contextlib import redirect_stdout



# Parameters
IMG_HEIGHT = 28
IMG_WIDTH = 28
PLATE_CHARS = 7
NUM_SAMPLES = 500
EPOCHS = 10
BATCH_SIZE = 32
SAVE_DIR = "./train_image"

os.makedirs(SAVE_DIR, exist_ok=True)

# Build dataset
X = []
y = []

for i in range(NUM_SAMPLES):
    plate_chars = np.random.choice(characters, PLATE_CHARS)
    plate_img = generate_license_plate_image(plate_chars)
    char_imgs = segment_plate_characters(plate_img)
    for j, (char, char_img) in enumerate(zip(plate_chars, char_imgs)):
        filename = f"char_{char}_{i:04d}_{j}.png"
        char_img.save(os.path.join(SAVE_DIR, filename))
        X.append(np.array(char_img))
        y.append(char_to_idx[char])

X = np.array(X).astype("float32") / 255.0
X = X.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
y = to_categorical(np.array(y), num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model

model = Sequential([
    Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax")
])
with redirect_stdout(sys.stderr):
    model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
with redirect_stdout(sys.stderr):
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

model.save("model.keras")

loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")


