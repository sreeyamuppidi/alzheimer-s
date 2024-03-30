import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = "DATASET/Alzheimer_s Dataset/train"
test_dir = "DATASET/Alzheimer_s Dataset/test"

# Data augmentation for training data
train_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_gen = train_augmentation.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # AlexNet input size
    batch_size=32,
    class_mode='categorical'
)

# Validation data generator
validation_augmentation = ImageDataGenerator(rescale=1./255)
validation_generator = validation_augmentation.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)

# Define the AlexNet model
model = Sequential()

# Layer 1
model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# Layer 2
model.add(Conv2D(256, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# Layer 3
model.add(Conv2D(384, (3, 3), activation='relu'))
model.add(BatchNormalization())

# Layer 4
model.add(Conv2D(384, (3, 3), activation='relu'))
model.add(BatchNormalization())

# Layer 5
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# Flatten
model.add(Flatten())

# Fully connected layers
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_gen, steps_per_epoch=10, epochs=30, verbose=1, validation_data=validation_generator)

# Save the model
model.save("model/alexnet.h5")

# Plot accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'], 'r', label='Training accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("model/alexnet_acc.png")
plt.show()

# Plot loss
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['loss'], 'r', label='Training Loss', color='green')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('# epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("model/alexnet_loss.png")
plt.show()
