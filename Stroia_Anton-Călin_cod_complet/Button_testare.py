import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 23
GPIO.set(BUTTON_PIN, GPIO.IN)

print("Astept sa apesi")

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            print("Buton neapasat")
        else:
            print("Buton apasat")
        time.sleep(0.2)
except KeyboardInterrupt:
    GPIO.cleanup()
    