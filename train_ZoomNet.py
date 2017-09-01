from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.initializers import RandomNormal, Constant
from keras.callbacks import ModelCheckpoint
import numpy as np
from time import time

# Define Batch Size
batch_size = 32

base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

for layer in base_model.layers:
    layer.trainable = False

# Creating the top model (Fully Connected Layers)
# top_model = Sequential()
# top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# top_model.add(Dense(4096, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(1, activation='sigmoid'))
# base_model.add(top_model


# Creating the top model (Fully Connected Layers)
x = base_model.output
x = Flatten(input_shape=base_model.output_shape[1:])(x)
x = Dense(4096, activation='relu', kernel_initializer=RandomNormal(stddev=0.0001), bias_initializer=Constant())(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_initializer=RandomNormal(stddev=0.0001), bias_initializer=Constant())(x)
x = Dropout(0.5)(x)
prediction = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant())(x)
model = Model(inputs=base_model.input, outputs=prediction)


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.01,momentum=0.9,decay=0.001),
              metrics=['accuracy'])

print("Model Loaded. Model Summary:")
print(model.summary())

# Defining Data Generators
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    directory='dataset/june_23_2017/train',
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    directory='dataset/june_23_2017/validation',
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='binary')

# Checkpoint Setup
filepath="weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the Model
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=1000//batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=500/batch_size,
    callbacks=callbacks_list
)

# time_image_load_start = time()
# img_path = 'dog.png'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# time_forward_pass_start = time()
# preds = base_model.predict(x)
# time_forward_pass_end = time()
# print('Time taken to process 1 image = {}'.format(time_forward_pass_end-time_forward_pass_start))
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds)[0])