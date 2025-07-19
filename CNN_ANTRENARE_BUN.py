import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Setări
img_height, img_width = 224, 224
batch_size = 16
epochs = 20

# Căi
train_dir = "dataset2/training"
test_dir = "dataset2/testing"

# Augmentare pentru train
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

# Doar rescalare pentru test
test_datagen = ImageDataGenerator(rescale=1./255)

# Generatoare
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True
)

val_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Model pre-antrenat (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # blocăm greutățile de bază

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks (doar EarlyStopping)
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True)
]

# Antrenare
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

# Salvăm modelul final DOAR în format .h5
model.save("tacamurii_model.h5")
