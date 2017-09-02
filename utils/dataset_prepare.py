# Code By Bharat Kambalur
# University of California, San Diego
# Date Created Aug 01 2017
# Date Last Modified Aug 25 2017

import os
import cv2
import numpy as np
from conf import dataset_conf as C

def process_section(x1, y1, x2, y2, gt_boxes_str):
    global im, im_name, scene, sec_counter, processed_data_folder

    roi_height = x2 - x1 + 1
    roi_width = y2 - y1 + 1
    ##########print roi_height, roi_width
    small_side = int(min([roi_width, roi_height]))
    if small_side >= C.MIN_SHORT_SIDE:
        o_a_percent = 0

        # cv2.imshow('Win', im[x1:x2 + 1, y1:y2 + 1])
        # cv2.waitKey(0)
        # cv2.destroyWindow('Win')

        if gt_boxes_str[0] == '':
            o_a_percent = 0
        else:
            for box_str in gt_boxes_str:
                if o_a_percent == 0:
                    #bx1, by1, bx2, by2 = map(int,[item.strip("(),") for item in box_str.split(" ")])
                    by1, bx1, by2, bx2 = map(int, [item.strip("(),") for item in box_str.split(" ")])
                    # ROI Area calculation
                    roi_a = (bx2-bx1)*(by2-by1)
                    # Overlap calculation
                    if (bx1<x2) and (by1<y2):
                        o_h = min([x2,bx2]) - max([bx1,x1]) # Overlap Height
                        o_w = min([y2, by2]) - max([by1, y1]) # Overlap Width
                        if o_h < 0: o_h = 0
                        if o_w < 0: o_w = 0
                        o_a = o_h * o_w # o_a -> Overlap area (in sq. pixels)
                    else:
                        o_a = 0
                    o_a_percent = int(round(o_a*100/roi_a))
                else:
                    break

        ##################################################################
        # print o_a_percent
        #cv2.rectangle(im, (by1, bx1), (by2, bx2), (255, 0, 0), 2)


        if o_a_percent >=C.MIN_ROI_OVERLAP_PERCENT:
            ###################################################################################
            #sec = im[x1:x2+1,y1:y2+1]

            # Save section as positive
            cv2.imwrite(processed_data_folder + scene + "_" +
                        im_name.split('.')[0] + "_" + str(sec_counter) + "_T"+".png", im[x1:x2+1,y1:y2+1])
            sec_counter = sec_counter + 1


        else:
            # Save section as negative
            cv2.imwrite(processed_data_folder + scene + "_" +
                        im_name.split('.')[0] + "_" + str(sec_counter) + "_F" + ".png", im[x1:x2 + 1, y1:y2 + 1])
            sec_counter = sec_counter + 1


        # Calculate center points to split
        h_half = int(roi_height / 2)
        w_half = int(roi_width / 2)
        x_c = x1 + h_half
        y_c = y1 + w_half
        # print (x1, y1, x_c - 1, y_c - 1)  # Top Left
        # print (x1, y_c, x_c - 1, y2)  # Top Right
        # print (x_c, y1, x2, y_c - 1)  # Bottom Left
        # print (x_c, y_c, x2, y2)  # Bottom Right
        # print (x_c - int(h_half / 2), y_c - int(w_half / 2), x_c + int(h_half / 2) - 1,
        #                 y_c + int(w_half / 2) - 1)
        process_section(x1, y1, x_c - 1, y_c - 1, gt_boxes_str)  # Top Left
        process_section(x1, y_c, x_c - 1, y2, gt_boxes_str)  # Top Right
        process_section(x_c, y1, x2, y_c - 1, gt_boxes_str)  # Bottom Left
        process_section(x_c, y_c, x2, y2, gt_boxes_str)  # Bottom Right
        process_section(x_c - int(h_half / 2), y_c - int(w_half / 2), x_c + int(h_half / 2) - 1,
                        y_c + int(w_half / 2) - 1, gt_boxes_str)



        # print bx1, by1, bx2, by2


def process_image(gt_str):
    global im
    gt_boxes_str = gt_str.split(") (")
    small_side = int(min(im.shape[0:2]))
    long_side = int(max(im.shape[0:2]))
    small_side_index = np.argmin(im.shape[0:2])
    long_side_index = np.argmax(im.shape[0:2])
    s_side_half = int(small_side/2)
    num_long_secs = int(round(long_side/s_side_half))
    long_sec_length = long_side / float(int(round(long_side/s_side_half)))
    points = map(int,[long_sec_length * n for n in xrange(num_long_secs)]) + [long_side]
    for point_n in range(len(points)-1):
        if small_side_index == 0:
            process_section(0, points[point_n], s_side_half - 1, points[point_n + 1]-1, gt_boxes_str)
            process_section(s_side_half, points[point_n], small_side - 1, points[point_n + 1]-1, gt_boxes_str)
        else:
            process_section(points[point_n], 0, points[point_n + 1] - 1, s_side_half - 1, gt_boxes_str)
            process_section(points[point_n], s_side_half, points[point_n + 1] - 1, small_side - 1, gt_boxes_str)



dataset_root_folder = "D:\Research\ZoomNet\dataset\june_23_2017"
processed_data_folder = "D:\Research\ZoomNet\dataset\june_23_2017\processed_images\\"

scene_folder_list = [dir for dir in os.listdir(dataset_root_folder) if (dir[0:5] == 'scene')]

for scene in scene_folder_list[6:]:
    print("Processing Scene: {}".format(scene))
    im_list = [im_name for im_name in os.listdir(dataset_root_folder + "\\" + scene) if (im_name[-3:] == 'png')]

    gt_file = open(dataset_root_folder + "\\" + scene + "\\" + "label.txt", "r")
    first_line_flag = False  # Flag to be set once first line is read. This line is for description only.
    gt_dict = {}
    for line in gt_file:
        if first_line_flag == False:
            first_line_flag = True
            continue
        gt_dict[(line.strip().split(':')[0])] = (line.strip().split(':')[1].strip())

    for im_name in im_list:

        im = cv2.imread(dataset_root_folder + "\\" + scene + "\\" + im_name)
        sec_counter = 0
        #process_section(0,0,im.shape[0],im.shape[1],gt_dict[im_name.split('.')[0]])
        #process_section(0, 0, 14, 14, gt_dict[im_name.split('.')[0]])
        process_image(gt_dict[im_name.split('.')[0]])
        print("Finished processing Image:" + im_name)
    print("Finished processing Scene:" + scene)
