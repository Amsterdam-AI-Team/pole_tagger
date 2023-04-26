# Imports
import cv2
import pathlib
import os
import math
import pandas as pd
import numpy as np
import xgboost as xgb
from PIL import Image


# Functions to adjust street light statistics
def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return np.abs(90 - math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))) * 180 / np.pi)


# Click event to click points and draw red line in image window
def click_event(event, x, y, flags, params):
    # Checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # Displaying the coordinates
        # on the image window
        refPt.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_single_axis, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)

        # Draw line on image between points
        if len(refPt) == 2:
            cv2.line(img_single_axis, (refPt[0][0], refPt[0][1]), (refPt[1][0], refPt[1][1]),
                     (0, 0, 255), 2)

        cv2.imshow('check single pole', img_single_axis)


# Function to run the click event and return the clicked coordinates
def run_click_event(in_file_ax):
    cv2.namedWindow("check single pole")
    cv2.moveWindow("check single pole", 250, 100)
    global refPt, img_single_axis
    refPt = []
    correct = False

    # Reading the image
    img_single_axis = cv2.imread(in_file_ax)
    width, height = Image.open(in_file_ax).size

    # Displaying the image
    cv2.imshow('check single pole', img_single_axis)

    # Setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('check single pole', click_event)

    # Wait for a key to be pressed to exit
    k = cv2.waitKey(0)
    if k == 13 or k == 32:
        cv2.destroyWindow("check single pole")
        return width, height, refPt
    else:
        cv2.destroyWindow("check single pole")
        w, h, rp = run_click_event(in_file_ax)
        return w, h, rp


# Function to adjust red line if not correct
def adjust_pole_statistics(in_folder, idx, obj):
    in_file_ax_x = str(
        list(pathlib.Path(in_folder + 'object_per_axis/x/').glob('{}_*'.format(idx)))[0])
    in_file_ax_y = str(
        list(pathlib.Path(in_folder + 'object_per_axis/y/').glob('{}_*'.format(idx)))[0])

    # Get global min and max in pole plot
    x_v = in_file_ax_x[:-4].split('_')
    y_v = in_file_ax_y[:-4].split('_')
    min_x, min_y, min_z = float(x_v[-4]), float(y_v[-4]), float(x_v[-3])
    max_x, max_y, max_z = float(x_v[-2]), float(y_v[-2]), float(x_v[-1])

    # Obtain new x,y,z in local coordinate system
    w_x, h_x, rp_ax_x = run_click_event(in_file_ax_x)
    w_y, h_y, rp_ax_y = run_click_event(in_file_ax_y)

    # Calculate distance per pixel local coordinate system
    pre_rd_x, pre_t_x = rp_ax_x[0][0], rp_ax_x[1][0]
    pre_rd_y, pre_t_y = rp_ax_y[0][0], rp_ax_y[1][0]
    pre_rd_z = np.mean([rp_ax_x[0][1], rp_ax_y[0][1]])
    pre_t_z = np.mean([rp_ax_x[1][1], rp_ax_y[1][1]])
    dis_per_pix_x = (max_x - min_x) / w_x
    dis_per_pix_y = (max_y - min_y) / w_y
    dis_per_pix_z = (max_z - min_z) / h_x

    # Calculate new x,y,z in global coordinate system
    rd_x = np.round(min_x + pre_rd_x * dis_per_pix_x, 2)
    rd_y = np.round(min_y + pre_rd_y * dis_per_pix_y, 2)
    rd_z = np.round(max_z - pre_rd_z * dis_per_pix_z, 2)
    t_x = np.round(min_x + pre_t_x * dis_per_pix_x, 2)
    t_y = np.round(min_y + pre_t_y * dis_per_pix_y, 2)
    t_z = np.round(max_z - pre_t_z * dis_per_pix_z, 2)

    # Calculate height and angle and adjust in object row
    rd = np.array([rd_x, rd_y, rd_z])
    t = np.array([t_x, t_y, t_z])
    height = np.round(length(t - rd), 2)
    pole_angle = np.round(angle((t - rd), np.array(t - [0, 0, t_z - rd_z])), 2)
    obj.rd_x, obj.rd_y, obj.z = rd_z = rd_x, rd_y, rd_z
    obj.tx, obj.ty, obj.tz = t_x, t_y, t_z
    obj.height = height
    obj.angle = pole_angle

    return obj


# Function to determine street light type
def determine_pole_type(type_classifier, obj):
    # Calculate probs street light types with regression model
    X = np.array([obj[['height', 'radius', 'm_r', 'm_g', 'm_b']]])
    probs = type_classifier.predict_proba(X)[0]
    idxs_probs = np.array(probs).argsort().tolist()[::-1]
    i = 0

    # Loop over optional street light types
    while i < len(idxs_probs):
        idxs_prob = idxs_probs[i]

        # Load street light type
        cv2.namedWindow("pole type")
        type_img = cv2.imread('data/images/types/{}.png'.format(str(idxs_prob)))
        h, w, _ = type_img.shape
        type_img = cv2.resize(type_img, (int(w * 0.7), int(h * 0.7)))
        cv2.imshow('pole type', type_img)

        # The function waitKey waits for a key event infinitely (when delay<=0)
        k = None
        while k not in [106, 102, 49, 127]:
            k = cv2.waitKey(0)
            if k == 106:  # [j] if type is correct
                obj['type'] = idxs_prob
                cv2.destroyWindow("pole type")
                return obj
            elif k == 102:  # [f] go to next type
                i += 1
            elif k == 49:  # [1] type is unkown
                obj['type'] = 99
                cv2.destroyWindow("pole type")
                return obj
            elif k == 127 and i != 0:  # [backspace] go to previous type
                i -= 1
            else:
                print('Key not valid...')

    obj['type'] = 99
    cv2.destroyWindow("pole type")
    return obj


# Pole tagger code to validate clusters found in segmented point clouds
def check_poles(in_folder, in_folder_imgs, csv_poles, out_file):
    pd.options.mode.chained_assignment = None
    cv2.namedWindow("check poles")
    idx = 0

    # Load model
    type_classifier = xgb.XGBClassifier()
    type_classifier.load_model('models/type_classifier.json')

    # Load csv data
    df_poles = pd.read_csv(in_folder + csv_poles)
    if os.path.isfile(in_folder + out_file):
        df_poles_adjusted = pd.read_csv(in_folder + out_file)
    else:
        df_poles_adjusted = df_poles
        df_poles_adjusted['code'] = -1
        df_poles_adjusted['type'] = -1
        df_poles_adjusted.to_csv(in_folder + out_file, index=False)

    # Loop over extracted segmentation examples
    while idx < len(df_poles):
        obj = df_poles.iloc[idx]
        if df_poles_adjusted.loc[idx, 'code'] < 0 or np.isnan(
                df_poles_adjusted.loc[idx, 'code']) == True:
            img_name = in_folder_imgs + 'object_all_axes/' + str(idx) + '.png'

            # Load segmenatation example
            if os.path.exists(img_name):
                img = cv2.imread(img_name)
                img = cv2.resize(img, (900, 400))
                cv2.moveWindow("check poles", 550, 100)
                cv2.imshow('check poles', img)

                # The function waitKey waits for a key event infinitely (when delay<=0)
                k, back = None, False
                while k not in [106, 99, 105, 110, 127, 27]:
                    k = cv2.waitKey(0)
                    if k == 106:  # [j]: next image
                        obj['code'] = 0
                        obj = determine_pole_type(type_classifier, obj)
                        continue
                    elif k == 99:  # [c] to correct pole
                        obj = adjust_pole_statistics(in_folder_imgs, idx, obj)
                        obj = determine_pole_type(type_classifier, obj)
                        obj['code'] = 1
                    elif k == 105:  # [i] false positive
                        obj['code'] = 2
                        continue
                    elif k == 110:  # [n] unclear/more than one pole
                        obj['code'] = 3
                        continue
                    elif k == 127 and idx > 0:  # [backspace] to go back one example
                        df_poles_adjusted.loc[idx - 1, 'code'] = -1
                        idx -= 2
                        back = True
                    elif k == 27:  # [esc] to exit the program
                        return 0
                    else:
                        print('Key not valid...')

                if back == False:
                    df_poles_adjusted.loc[idx] = obj
                df_poles_adjusted.to_csv(in_folder + out_file, index=False)

        idx += 1
        if idx < 0:
            idx = 0


if __name__ == "__main__":
    in_out_folder_csv = 'data/csv_files/'
    in_folder_images = 'data/images/objects/'
    csv_poles = 'extracted_poles.csv'
    out_file = 'extracted_poles_checked.csv'
    check_poles(in_out_folder_csv, in_folder_images, csv_poles, out_file)
