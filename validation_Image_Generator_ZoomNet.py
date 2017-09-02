from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
import numpy as np
from time import time

batch_size = 64

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

model.load_weights('ZoomNet_Weights_100.hdf5')

print("Model Loaded. Model Summary:")
print(model.summary())


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    directory='dataset/june_23_2017/validation/false',
    target_size=(224,224),
    batch_size=batch_size,
    shuffle = True,
    class_mode=None)

#time_image_load_start = time()

#time_forward_pass_start = time()
preds = model.predict_generator(test_generator, steps = 100, verbose = 1)
#time_forward_pass_end = time()
#print('Time taken to process 1 image = {}'.format(time_forward_pass_end-time_forward_pass_start))
# decode the results into a list of tuples (class, description, probability)
#(one such list for each sample in the batch)
thresh = 0.4
num_true = np.sum(preds.flatten()<thresh)
#print('Predicted: {}'.format(preds))
print(num_true)
