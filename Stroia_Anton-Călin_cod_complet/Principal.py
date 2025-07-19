import subprocess
import time
import os
import RPi.GPIO as GPIO
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import numpy as np
import tflite_runtime.interpreter as tflite

BUTTON_PIN = 18
model_path = "tacamuri_model.tflite"
class_names = ['fork','knife','plate','spoon']
img_height, img_width = 224, 224
captura_path = "captura.jpg"

#Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#Load model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\Sistemul e gata")

def capture_image():
    subprocess.run(["libcamera-still", "-o", captura_path, "-t", "1000", "--width", "4608", "--height", "2592", "--nopreview",])

def load_and_preprocess_image():
    img = Image.open(captura_path).convert("RGB").resize((img_width, img_height))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_array):
    interpreter.set_tensor(input_details[0]['index'],img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_index = int(np.argmax(output_data))
    confidence = float(output_data[0][class_index])
    return class_index,confidence

def feedback_vocal(mesaj):
    os.system(f'espeak "{mesaj}"')

def verifica_cadru(img):
    width, height = img.size
    brightness = np.mean(np.array(img))
    if brightness < 50:
        return "Prea întunecat, adu mai aproape de lumina"
    if brightness > 230:
        return "Prea luminos, ajusteaza pozitia"
    return "OK"

def detect_text(img):
    gray_image = ImageOps.grayscale(img)
    gray_image = gray_image.filter(ImageFilter.MedianFilter(size=3))
    enhanced = ImageOps.autocontrast(gray_image)
    treshold = 0.5
    bw= enhanced.point(lambda x:0 if x<255 * treshold else 255, '1')
    custom_config = '--psm 6'
    text = pytesseract.image_to_string(bw, lang='ron',config=custom_config)
    return text.strip()

mode = "object"
print("Pornit in modul de detectie obiecte")

while True:
    if mode == "object":
        capture_image()
        img = Image.open(captura_path)
        status = verifica_cadru(img)

        if status !="OK":
            print(f"Problema: {status}")
            feedback_vocal(status)
            time.sleep(2)
            continue

        img_array = load_and_preprocess_image()
        class_index, confidence = predict(img_array)

        if confidence > 0.7:
            result_text = f"Obiectul este {class_names[class_index]} cu incredere de {confidence:.2f}"
            print(result_text)
            feedback_vocal(f"Am detectat {class_names[class_index]}")
        else:
            feedback_vocal("Nu detectez clar obiectul, ajuteaza pozitia")

        time.sleep(2)

        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            mode = "text"
            print("Mod schimbat la detectie text")
            feedback_vocal("Modul detectare text activ")
            time.sleep(1)
    elif mode == "text":
        capture_image()
        img = Image.open(captura_path)
        text = detect_text(img)
        print(f"Text detectat:\n{text}")
        if text:
            feedback_vocal(text)
        else:
            feedback_vocal("Nu am detectat text")

######ASTEAPTA APASARE BUTON CA SA REVINA IN MODUL DE OBIECTE
        print("Astept sa fie apasat butonul pentru a reveni la detectare de obiecte")
        feedback_vocal("Apasa butonul pentru a reveni la detectarea obiectelor")

        while True:
            if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
                mode = "object"
                print("Mod schimbat la detectie de obiecte")
                feedback_vocal("Modul detecare obiecte activat")
                time.sleep(1)
                break
        time.sleep(1)
