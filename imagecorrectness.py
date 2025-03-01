import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

train_path = r"C:\Users\Kasey\OneDrive\Documents\sih\image correction analysis main\ml\images\train"
validation_path = r'C:\Users\Kasey\OneDrive\Documents\sih\image correction analysis main\ml\images\val'
test_path = r'C:\Users\Kasey\OneDrive\Documents\sih\image correction analysis main\ml\images\val'

image_shape = (128, 128, 3)  # Adjust based on your dataset
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(image_shape[0], image_shape[1]),
    batch_size=batch_size,
    class_mode='binary'  # Assuming binary classification (complete or not)
)

validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(image_shape[0], image_shape[1]),
    batch_size=batch_size,
    class_mode='binary'
)

def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

cnn_model = create_cnn_model(image_shape)
compile_model(cnn_model)

history = cnn_model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=10,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples // batch_size)

test_generator = validation_datagen.flow_from_directory(
    test_path,
    target_size=(image_shape[0], image_shape[1]),
    batch_size=batch_size,
    class_mode='binary'
)

test_loss, test_accuracy = cnn_model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy}')

# Assuming you already have your model, test_generator, and image_shape defined

# Load and preprocess test data
test_generator = validation_datagen.flow_from_directory(
    test_path,
    target_size=(image_shape[0], image_shape[1]),
    batch_size=batch_size,
    class_mode='binary',  # Update class_mode based on your task
    shuffle=False  # Set shuffle to False for accurate evaluation
)

# Evaluate the model on the test set
test_loss, test_accuracy = cnn_model.evaluate(test_generator)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

cnn_model.save('cnnmodel.h5')

'''
import numpy as np
from keras.preprocessing import image

# Function to load and preprocess an image for testing
def load_and_preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Path to the image you want to test
test_image_path = '/content/WhatsApp Image 2023-12-09 at 7.46.56 PM.jpeg'

# Load and preprocess the test image
test_image = load_and_preprocess_image(test_image_path, target_size=(image_shape[0], image_shape[1]))

# Make predictions
predictions = cnn_model.predict(test_image)

# Get the predicted class (binary classification)
predicted_class = 1 if predictions[0, 0] > 0.5 else 0

# Print the prediction
print(f'Predicted Class: {predicted_class}')'''