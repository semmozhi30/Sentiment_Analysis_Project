# train_cnn_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib
import os

# Directories
train_dir = 'fer2013/train'
test_dir = 'fer2013/test'

# Image size and parameters
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 10

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=EPOCHS, validation_data=test_data)

# Save model
model.save('model/expression_model.h5')
print('✅ Model saved to model/expression_model.h5')

# Save label index
os.makedirs('model', exist_ok=True)
joblib.dump(train_data.class_indices, 'model/label_map.pkl')
print('✅ Class label map saved to model/label_map.pkl')
