from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from time import time

model = VGG16(weights='imagenet', include_top=True)

time_image_load_start = t
img_path = 'dog.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

time_forward_pass_start = time()
preds = model.predict(x)
time_forward_pass_end = time()
print('Time taken to process 1 image = {}'.format(time_forward_pass_end-time_forward_pass_start))
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds)[0])