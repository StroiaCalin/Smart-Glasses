import tensorflow as tf
import numpy as np
import cv2

# Încarcă modelul antrenat
model = tf.keras.models.load_model('tacamurii_model.h5')

# Setări imagine
img_height, img_width = 224, 224
class_names = ['fork', 'knife', 'plate', 'spoon']

# Pornește camera (0 = camera principală)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Eroare: nu pot accesa camera.")
    exit()

print("✅ Camera pornită. Apasă 'q' pentru a ieși.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Eroare la citirea frame-ului.")
        break

    img = cv2.resize(frame, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0) / 255.0  # normalizare

    predictions = model.predict(img)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100

    if confidence >= 90:
        label = f"{predicted_class} ({confidence:.2f}%)"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Tacamuri Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Curățare resurse
cap.release()
cv2.destroyAllWindows()
