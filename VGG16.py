import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten,BatchNormalization,Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pylab as plt

train_dir="DATASET/Alzheimer_s Dataset/train"
test_dir="DATASET/Alzheimer_s Dataset/test"
train_augmentation = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,
                                    zoom_range = 0.2,horizontal_flip = True,)

train_gen = train_augmentation.flow_from_directory(train_dir,
                                                target_size=(128,128),
                                                batch_size=12,
                                                class_mode='categorical')

validation_augmentation=ImageDataGenerator(rescale=1./255)
validation_generator = validation_augmentation.flow_from_directory(test_dir,
                                                target_size=(128,128),
                                                batch_size=12,
                                                class_mode='categorical',
                                               )

conv_base=tf.keras.applications.VGG16(input_shape=(128,128,3),include_top=False,weights='imagenet')

conv_base.summary()

for layer in conv_base.layers:
    layer.trainable=False

model=Sequential()

model.add(conv_base)
model.add(BatchNormalization())

model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))
model.add(Dropout(0.2))

model.add(layers.Dense(4,activation='softmax'))

model.summary()

conv_base.trainable=True
set_trainable=False
for layer in conv_base.layers:
    if layer.name=='blocks_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_gen,
                  steps_per_epoch=10,
                  epochs=30,
                  verbose=1,
                  validation_data=validation_generator)

model.save(r"model\vgg.h5")
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='Training accuracy',color='green')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(r"model\vgg_acc.png")
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='Training Loss',color='green')
plt.plot(history.history['val_accuracy'],label='validation Loss')
plt.xlabel('# epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(r"model\vgg_loss.png")
plt.show()


