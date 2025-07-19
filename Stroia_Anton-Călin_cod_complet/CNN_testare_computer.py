import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("tacamuri_model.h5")

class_names = ['fork', 'knife', 'plate', 'spoon']

img_height, img_width = 224, 224

cap = cv2.VideoCapture(0) 

print("Apasă [SPACE] pentru a face o poză și a prezice.")
print("Apasă [ESC] pentru a ieși.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == 32:
        img_resized = cv2.resize(frame, (img_width, img_height))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        confidence = prediction[0][class_index]

        print(f"Predicție: {class_name} ({confidence:.2f})")

        cv2.putText(frame, f"{class_name} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Predicție", frame)
        cv2.waitKey(0) 

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
