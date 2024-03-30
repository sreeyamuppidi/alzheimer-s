import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Activation,Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Flatten, Dense, Dropout, BatchNormalization

img_height,img_width=(128,128)

batch_size=12
train_data_dir="DATASET/Alzheimer_s Dataset/train"
test_data_dir="DATASET/Alzheimer_s Dataset/test"
train_datagen = ImageDataGenerator(shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True,validation_split=0.4)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                target_size=(img_height,img_width),
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                subset='validation')
test_generator = train_datagen.flow_from_directory(test_data_dir,
                                                target_size=(img_height,img_width),
                                                batch_size=1,
                                                class_mode='categorical',
                                                subset='validation')
x,y=test_generator.next()

# (3) Create a sequential model
base_model = tf.keras.applications.MobileNet(input_shape=(128, 128, 3), include_top=False,
                          weights='imagenet')
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(4, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_generator,epochs=50,validation_data=test_generator)
model.save(r"model\Mobilenet.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='training accuracy',color='green')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(r"model\mobilenet_acc.png")
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['loss'],'r',label='training loss',color='green')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(r"model\mobilenet_loss.png")
plt.show()