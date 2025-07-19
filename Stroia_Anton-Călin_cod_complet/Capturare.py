import os
import time
num_poze = 50

folder = "/home/calin/Desktop/spoon"

for i in range (1, num_poze + 1):
    filename = f"{folder}/capture{i}.jpg"
    command = f"libcamera-still -t 1000 -n -o {filename}"
    print(f"Capturand poza {i}/{num_poze}")
    os.system(command)
    time.sleep(1)

print("Toate pozele au fost realizate")
