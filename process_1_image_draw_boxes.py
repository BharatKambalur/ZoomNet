# Code By Bharat Kambalur
# University of California, San Diego
# Date Created Sep 01 2017
# Date Last Modified Sep 04 2017

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
import numpy as np
from time import time
import cv2

# TODO: Add confirguration file

im_path = "test.png"
weights_file = "weights//ZoomNet_Weights_100.hdf5"

def process_section(x1, y1, x2, y2, sec_id):
    global img_pre, global_s_id
    global im_list_to_process, subsec_bounds_list, sec_subsec_list

    sec_height = x2 - x1 + 1
    sec_width = y2 - y1 + 1
    small_side = int(min([sec_width, sec_height]))
    if small_side >= 40: #TODO: Add configuration parameter for minimum short side length
        # Store the section of the image to the global list to be fed through the network and store the bounding
        # coordinates in the list as well. Also, add the subsection ID to the corresponding section ID list.
        im_sec_reshaped = np.expand_dims(cv2.resize(img_pre[x1:x2+1,y1:y2+1],(224,224)), axis=0)
        im_list_to_process = np.vstack((im_list_to_process, im_sec_reshaped))  # Add image to queue
        subsec_bounds_list = np.vstack((subsec_bounds_list,[x1, y1, x2, y2]))
        sec_subsec_list[sec_id].append(global_s_id)
        global_s_id = global_s_id + 1

        # Calculate center points to split
        h_half = int(sec_height / 2)
        w_half = int(sec_width / 2)
        x_c = x1 + h_half
        y_c = y1 + w_half
        # print (x1, y1, x_c - 1, y_c - 1)  # Top Left
        # print (x1, y_c, x_c - 1, y2)  # Top Right
        # print (x_c, y1, x2, y_c - 1)  # Bottom Left
        # print (x_c, y_c, x2, y2)  # Bottom Right
        # print (x_c - int(h_half / 2), y_c - int(w_half / 2), x_c + int(h_half / 2) - 1,
        #                 y_c + int(w_half / 2) - 1)
        process_section(x1, y1, x_c - 1, y_c - 1, sec_id)  # Top Left
        process_section(x1, y_c, x_c - 1, y2, sec_id)  # Top Right
        process_section(x_c, y1, x2, y_c - 1, sec_id)  # Bottom Left
        process_section(x_c, y_c, x2, y2, sec_id)  # Bottom Right
        process_section(x_c - int(h_half / 2), y_c - int(w_half / 2), x_c + int(h_half / 2) - 1,
                        y_c + int(w_half / 2) - 1, sec_id)  # Center Section


        # print bx1, by1, bx2, by2


def split_image_to_array():
    global sec_id, global_s_id, sec_subsec_list
    small_side = int(min(img_pre.shape[0:2]))
    long_side = int(max(img_pre.shape[0:2]))
    small_side_index = np.argmin(img_pre.shape[0:2])
    s_side_half = int(small_side/2)
    num_long_secs = int(round(long_side/s_side_half))
    long_sec_length = long_side / float(num_long_secs)
    l_sec_half = int(int(long_sec_length)/2)
    s_sec_half = int(s_side_half/2)
    points = list(map(int,[long_sec_length * n for n in range(num_long_secs)])) + [long_side]
    sec_id = -1      # Reset the global section and subsecttion counters to 0 for a new image
    global_s_id = 0
    for point_n in range(len(points)-1):
        if small_side_index == 0:
            sec_id = sec_id + 1
            sec_subsec_list.append([])
            process_section(0, points[point_n], s_side_half - 1, points[point_n + 1]-1, sec_id)  # Top Row
            sec_id = sec_id + 1
            sec_subsec_list.append([])
            process_section(s_side_half, points[point_n], small_side - 1, points[point_n + 1]-1, sec_id)  # Bottom Row
            if point_n != 0:            # Overlapping middle sections
                sec_id = sec_id + 1
                sec_subsec_list.append([])
                process_section(s_side_half-s_sec_half+1, points[point_n]-l_sec_half,
                                s_side_half+s_sec_half, points[point_n]+l_sec_half-1, sec_id)
        else:
            sec_id = sec_id + 1
            sec_subsec_list.append([])
            process_section(points[point_n], 0, points[point_n + 1] - 1, s_side_half - 1, sec_id)  # Top Row
            sec_id = sec_id + 1
            sec_subsec_list.append([])
            process_section(points[point_n], s_side_half, points[point_n + 1] - 1, small_side - 1, sec_id)  # Bottom Row
            if point_n != 0:            # Overlapping middle sections
                sec_id = sec_id + 1
                sec_subsec_list.append([])
                process_section(points[point_n] - l_sec_half, s_side_half - s_sec_half,
                                points[point_n] + l_sec_half - 1, s_side_half + s_sec_half - 1, sec_id)


def draw_rectangle_global_image(coord):
    global img_cv2_arr
    col = (224, 29, 68)
    border_col = (0, 204, 255)
    alpha = 0.1
    overlay = img_cv2_arr.copy()
    output = img_cv2_arr.copy()
    cv2.rectangle(overlay, (coord[1], coord[0]), (coord[3], coord[2]), col, -1)
    cv2.rectangle(overlay, (coord[1], coord[0]), (coord[3], coord[2]), border_col, 5)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, img_cv2_arr)

#### LOAD IMAGE AND PREPROCESS
time_image_load_start = time()
img = image.load_img(im_path)  # Loaded image is in PIL format
img_cv2_arr = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)  # Converting to CV2 Format to draw rectangles
img_arr = image.img_to_array(img)  # Conversion to array used for Keras
img_pre = preprocess_input(img_arr)
time_image_load_end = time()
print("Image {0} Loaded in time {1}s".format(im_path,time_image_load_end-time_image_load_start))

## Global Variable declarations
sec_id = 0 # ID for each section (top=level region in the whole image) (usually 4 or 6 sections per image)
global_s_id = 0 # ID to track each section and subsection. Used to queue up all parts of the image for batch prediction
subsec_bounds_list = np.empty((0,4),dtype=int) # List which stores the boundaries of the subsections that are processed
sec_subsec_list = [] # List which stores the corresponding subsections for each section
im_list_to_process = np.empty((0,224,224,3)) # 4 X n List which stores the 'n' images to be fed through the network

#### SPLIT LOADED IMAGE INTO ARRAY
print("")
#img_pre = img_pre[0:17,0:24]
split_image_to_array()

#### DEFINE AND LOAD MODEL
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

model.load_weights(weights_file)

print("Model Created. Model Summary:")
print(model.summary())
print("Weights loaded from file: {}".format(weights_file))

processed_op = model.predict(im_list_to_process)
th = 0.1
for section_n in range(len(processed_op)):
    if processed_op[section_n]>th:
        print processed_op[section_n]
        draw_rectangle_global_image(subsec_bounds_list[section_n])


# for main_sec in sec_subsec_list:
#     print(processed_op[main_sec[0]])
#     draw_rectangle_global_image(subsec_bounds_list[main_sec[0]])
#     cv2.imshow('Final Output', img_cv2_arr)
#     cv2.waitKey(0)


print(im_path.split(".")[0]+"_p.png")
cv2.imwrite(im_path.split(".")[0]+"_p.png",img_cv2_arr)
cv2.imshow('Final Output',img_cv2_arr)
cv2.waitKey(0)