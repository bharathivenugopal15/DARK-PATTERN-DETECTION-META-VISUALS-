
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Define constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2  # Adjust according to your dataset size
EPOCHS = 10

# Data preparation - Assuming 'train' and 'val' are your subfolders
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\SHARMILA\Documents\My documents\image correction analysis\ml\altered_train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    r'C:\Users\SHARMILA\Documents\My documents\image correction analysis\ml\altered_val',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Define a simplified CNN model due to the small dataset
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the best model during training
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test Accuracy: {test_acc}')

# Save the model
model.save('alteration_model.h5')

import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Function to load and preprocess an image for testing
def load_and_preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

'''
# Path to the image you want to test
test_image_path = '/content/drive/MyDrive/alteredmain/val/not altered/correct17.jpg'

# Assuming you have defined image_shape somewhere
image_shape = (256, 256)  # Replace with the actual shape used during training

# Load and preprocess the test image
test_image = load_and_preprocess_image(test_image_path, target_size=image_shape)

# Load the trained CNN model
model_path = 'image_alteration_detection_model.h5'  # Replace with the actual path to your model file
cnn_model = load_model(model_path)

# Make predictions
predictions = cnn_model.predict(test_image)

# Get the predicted class (binary classification)
predicted_class = 1 if predictions[0, 0] > 0.5 else 0

# Print the prediction
print(f'Predicted Class: {predicted_class}')'''