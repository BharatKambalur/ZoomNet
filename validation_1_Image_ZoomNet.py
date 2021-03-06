from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
import numpy as np
from time import time

base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

for layer in base_model.layers:
    layer.trainable = False

# Creating the top model (Fully Connected Layers)
x = base_model.output
x = Flatten(input_shape=base_model.output_shape[1:])(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=prediction)


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.01,momentum=0.9,decay=0.001),
              metrics=['accuracy'])

model.load_weights('weights/ZoomNet_Weights_100.hdf5')

print("Model Loaded. Model Summary:")
print(model.summary())


time_image_load_start = time()
img_path = 'dataset/june_23_2017/validation/true/scene6_00301_95_T.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


img_path = 'dataset/june_23_2017/validation/true/scene6_00221_105_T.png'
img = image.load_img(img_path, target_size=(224, 224))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)
y = preprocess_input(y)


z = np.concatenate((x,y))

time_forward_pass_start = time()
preds = model.predict_on_batch(z)
time_forward_pass_end = time()
print('Time taken to process 1 image = {}'.format(time_forward_pass_end-time_forward_pass_start))
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted: {}'.format(preds))
