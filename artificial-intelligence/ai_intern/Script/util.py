from PIL import Image, ImageDraw, ImageFont, ImageOps
from tensorflow.keras.models import load_model
import random
import string
import numpy as np

characters = list(string.ascii_uppercase + string.digits)
num_classes = len(characters)
char_to_idx = {char: idx for idx, char in enumerate(characters)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

FONT_PATHS = [
    "./Fonts/Arial.ttf",
    "./Fonts/Verdana.ttf",
    "./Fonts/Georgia.ttf",
    "./Fonts/TimesNewRoman.ttf",
    "./Fonts/Tahoma.ttf",
    "./Fonts/CourierNew.ttf",
]
def generate_char_image(char, char_size=28):
    font_images = []
    for font_path in FONT_PATHS:
        try:
            font = ImageFont.truetype(font_path, int(char_size * 0.8))
        except:
            font = ImageFont.load_default()
        img = Image.new("L", (char_size, char_size), 255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        offset_x = (char_size - w) // 2
        offset_y = (char_size - h) // 2
        draw.text((offset_x, offset_y), char, font=font, fill=0)
        font_images.append((font_path, img))
    return font_images

def generate_license_plate_image(chars, char_size=28):
    width = char_size * len(chars)
    plate_img = Image.new("L", (width, char_size), 255)
    for i, c in enumerate(chars):
        _, char_img = random.choice(generate_char_image(c, char_size))
        plate_img.paste(char_img, (i * char_size, 0))
    return plate_img

def segment_plate_characters(plate_img, num_chars=7, char_size=28):
    plate_img = plate_img.resize((num_chars * char_size, char_size))
    return [plate_img.crop((i * char_size, 0, (i + 1) * char_size, char_size)) for i in range(num_chars)]


def resize_plate_to_fixed_size(plate_img, target_height=28, num_chars=7):
    """
    Resize a plate image of any size to a fixed size (target_height x char_width*num_chars),
    preserving aspect ratio and padding width if necessary.

    Args:
        plate_img (PIL.Image): Input plate image (grayscale).
        target_height (int): Desired height (e.g., 28).
        num_chars (int): Number of characters on the plate.

    Returns:
        PIL.Image: Resized and padded plate image of size (target_height, target_height * num_chars).
    """
    target_width = target_height * num_chars
    # Resize plate height to target_height, maintain aspect ratio
    w, h = plate_img.size
    new_width = int(w * (target_height / h))
    resized = plate_img.resize((new_width, target_height), Image.Resampling.LANCZOS)

    # Pad width if less than target_width
    if new_width < target_width:
        delta_w = target_width - new_width
        padding = (0, 0, delta_w, 0)  # pad on right side
        resized = ImageOps.expand(resized, padding, fill=255)  # white padding

    # If width > target_width, crop centered to target_width
    elif new_width > target_width:
        left = (new_width - target_width) // 2
        resized = resized.crop((left, 0, left + target_width, target_height))

    return resized

def predict_plate_chars(plate_img_path, num_chars=7, model_path="model.keras"):
    # Load model (you might want to load it once outside this function if calling many times)
    model = load_model(model_path)

    # Open image and convert to grayscale
    plate_img = Image.open(plate_img_path).convert("L")

    # Resize plate to fixed size (28 height, 28*num_chars width)
    plate_img = resize_plate_to_fixed_size(plate_img, target_height=28, num_chars=num_chars)

    # Segment plate into character images
    char_imgs = segment_plate_characters(plate_img, num_chars=num_chars, char_size=28)

    predicted_text = ""
    for char_img in char_imgs:
        # Normalize and reshape for model
        img_array = np.array(char_img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict character and append
        pred = model.predict(img_array, verbose=0)
        predicted_text += idx_to_char[np.argmax(pred)]

    return predicted_text