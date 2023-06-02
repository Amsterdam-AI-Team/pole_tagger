"""Street light annotation tool for type and fit"""
# Imports
import argparse
import ast
import collections
import copy
import math
import os
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd
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
        cv2.putText(img_single_axis, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)

        # Draw line on image between points
        if len(refPt) == 2:
            cv2.line(
                img_single_axis,
                (refPt[0][0], refPt[0][1]),
                (refPt[1][0], refPt[1][1]),
                (0, 0, 255),
                2,
            )

            cv2.setMouseCallback("check single pole", lambda *args: None)

        cv2.imshow("check single pole", img_single_axis)

    elif event == cv2.EVENT_MOUSEMOVE:
        if len(refPt) == 1:
            temp_img = copy.deepcopy(img_single_axis)
            cv2.line(
                temp_img,
                (refPt[0][0], refPt[0][1]),
                (x, y),
                (0, 0, 255),
                1,
            )

            cv2.imshow("check single pole", temp_img)


# Function to run the click event and return the clicked coordinates
def run_click_event(in_file_ax, window_name="check single pole"):
    cv2.namedWindow("check single pole")
    cv2.moveWindow("check single pole", 450, 100)
    cv2.setWindowTitle("check single pole", window_name)

    global refPt, img_single_axis
    refPt = []

    # Reading the image
    img_single_axis = cv2.imread(in_file_ax)
    width, height = Image.open(in_file_ax).size
    width, height = width * 1.5, height * 1.5

    # Displaying the image
    hei, wid, _ = img_single_axis.shape
    img_single_axis = cv2.resize(img_single_axis, (int(wid * 1.5), int(hei * 1.5)))
    cv2.imshow("check single pole", img_single_axis)

    # Setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback("check single pole", click_event)

    # Wait for a key to be pressed to exit
    k = cv2.waitKey(0)
    cv2.destroyWindow("check single pole")
    if k == 27:  # [esc]: escape the program
        return 0, 0, 0
    elif k == 102 or k == 50:  # [f] or [2]: not a street light
        return -1, -1, -1
    elif k == 13 or k == 32:  # [space] or [enter]: fit is correct
        return width, height, refPt
    else:  # any other key: fit is incorrect, do refit
        w, h, rp = run_click_event(in_file_ax, window_name)
        return w, h, rp


# Function to adjust red line if not correct
def adjust_pole_statistics(in_folder, img_number, obj):
    in_file_ax = {
        axis: str(
            list(pathlib.Path(in_folder + f"object_per_axis/{axis}/").glob("{}_*".format(img_number)))[0]
        )
        for axis in ["x", "y"]
    }

    # Get global min and max in pole plot
    x_v = in_file_ax["x"][:-4].split("_")
    y_v = in_file_ax["y"][:-4].split("_")
    min_x, min_y, min_z = float(x_v[-4]), float(y_v[-4]), float(x_v[-3])
    max_x, max_y, max_z = float(x_v[-2]), float(y_v[-2]), float(x_v[-1])

    w, h, rp_ax = {}, {}, {}

    # Obtain new x,y,z in local coordinate system
    for axis in ["x", "y"]:
        w[axis], h[axis], temp_rp_ax = run_click_event(
            in_file_ax[axis], f"Adjust pole {img_number} ({axis})"
        )
        if w[axis] == 0:  # check if program was escaped
            return obj, True, 0
        elif w[axis] == -1:  # check if not a clear street light
            return obj, False, -2

        # Swap to bottom -> top point independent of click order
        rp_ax[axis] = (
            temp_rp_ax if temp_rp_ax[0][1] < temp_rp_ax[1][1] else [temp_rp_ax[1], temp_rp_ax[0]]
        )

    # Calculate distance per pixel local coordinate system
    pre_rd_x, pre_t_x = rp_ax["x"][0][0], rp_ax["x"][1][0]
    pre_rd_y, pre_t_y = rp_ax["y"][0][0], rp_ax["y"][1][0]
    pre_rd_z = np.mean([rp_ax["x"][1][1], rp_ax["y"][1][1]])
    pre_t_z = np.mean([rp_ax["x"][0][1], rp_ax["y"][0][1]])
    dis_per_pix_x = (max_x - min_x) / w["x"]
    dis_per_pix_y = (max_y - min_y) / w["y"]
    dis_per_pix_z = (max_z - min_z) / h["x"]

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

    return obj, False, 0


# Function to calculate type probabilities
def get_pole_type_probs(type_classifier, obj):
    # Calculate probs street light types with regression model
    X = np.array([obj[["height", "radius", "m_r", "m_g", "m_b"]]])
    probs = type_classifier.predict_proba(X)[0]
    idxs_probs = np.array(probs).argsort().tolist()[::-1]
    obj["type_preds"] = str(idxs_probs)
    obj["type_probs"] = str(sorted(probs, reverse=True))
    return obj


# Function to iterate over street light types
def determine_pole_type(preds, obj):  # noqa: C901
    # Load street light type
    i = 0
    while i < len(preds):
        pred = preds[i]
        cv2.namedWindow("pole type")
        cv2.moveWindow("pole type", 50, 50)
        cv2.setWindowTitle("pole type", f"Pole type {pred + 1}")
        type_img = cv2.imread("data/images/types/{}.png".format(str(pred)))
        h, w, _ = type_img.shape
        type_img = cv2.resize(type_img, (int(w * 0.7), int(h * 0.7)))
        cv2.imshow("pole type", type_img)

        # The function waitKey waits for a key event infinitely (when delay<=0)
        k = None
        while k not in [2, 3, 13, 27, 32, 49, 81, 83, 106, 127]:
            k = cv2.waitKey(0)
            if k == 13:  # [enter]: type is correct
                obj["type"] = pred
                cv2.destroyWindow("pole type")
                return obj, False
            elif k == 3 or k == 83:  # [>]: go to next type
                i += 1
            elif k == 49 or k == 117:  # [1] or [u] type is unknown
                obj["type"] = 99
                cv2.destroyWindow("pole type")
                return obj, False
            elif k in [2, 81] and i != 0:  # [<]: go to previous type
                i -= 1
            elif k == 8 or k == 127:  # [backspace]: go to previous pole to determine type
                return obj, "back"
            elif k == 27:  # [escape]: exit the program
                return obj, True
            else:
                print("Key not valid...")
            if i == len(preds):
                i -= 1


# Function to validate pole clusters found in segmented point clouds
def validate_poles(in_folder_imgs, csv_poles, out_file):  # noqa: C901
    pd.options.mode.chained_assignment = None
    cv2.namedWindow("check poles")
    idx = 0

    # Load csv data
    df_poles = pd.read_csv(csv_poles)
    if os.path.isfile(out_file):
        df_poles_adjusted = pd.read_csv(out_file)
        df_poles_adjusted["type"] = df_poles_adjusted["type"].astype("Int64")
    else:
        df_poles_adjusted = df_poles
        df_poles_adjusted["code"] = -1
        df_poles_adjusted["type"] = None
        df_poles_adjusted["type_preds"] = None
        df_poles_adjusted["type_probs"] = None
        df_poles_adjusted["type_preds"] = df_poles_adjusted["type_preds"].astype("object")
        df_poles_adjusted["type_probs"] = df_poles_adjusted["type_probs"].astype("object")
        df_poles_adjusted.to_csv(out_file, index=False)

    # Load model
    type_classifier = xgb.XGBClassifier()
    type_classifier.load_model("models/type_classifier.json")

    # Loop over extracted segmentation examples
    while idx < len(df_poles):
        obj = df_poles.iloc[idx]
        if df_poles_adjusted.loc[idx, "code"] == -1:
            img_number = str(df_poles_adjusted.loc[idx, 'index'])
            img_name = in_folder_imgs + "object_all_axes/" + img_number + ".png"

            # Load segmentation example
            if os.path.exists(img_name):
                img = cv2.imread(img_name)
                cv2.moveWindow("check poles", 275, 100)
                cv2.setWindowTitle("check poles", f"Check pole {img_number}")
                cv2.imshow("check poles", img)

                # The function waitKey waits for a key event infinitely (when delay<=0)
                k, back = None, False
                pos_values = [13, 32, 3, 99, 102, 117, 8, 127, 2, 49, 50, 51, 52, 81, 83]
                while k not in pos_values:
                    k = cv2.waitKey(0)
                    if k in [13, 32, 3, 83, 49]:
                        # [enter] or [space] or [>] or [1]: true positive
                        obj["code"] = 0
                        obj = get_pole_type_probs(type_classifier, obj)
                        continue
                    elif k == 102 or k == 50:  # [f] or [2]: false positive
                        obj["code"] = 2
                        continue
                    elif k == 99 or k == 51:  # [c] or [3]: pole needs correction
                        obj["code"] = 1
                    elif k == 117 or k == 52:  # [u] or [4]: unclear/more than one pole
                        obj["code"] = 3
                        continue
                    elif (
                        k in [8, 127, 2, 81] and idx > 0
                    ):  # [backspace] or [<]: go back one example
                        df_poles_adjusted.loc[idx - 1, "code"] = -1
                        idx -= 2
                        back = True
                    elif k == 27:  # [esc] to exit the program
                        return 0
                    else:
                        print("Key not valid...")

                if back is False:
                    df_poles_adjusted.loc[idx] = obj
                df_poles_adjusted.to_csv(out_file, index=False)

        idx += 1
        if idx < 0:
            idx = 0

    print("There are no more poles to validate.")


# Function to adjust fit of street light
def adjust_fit(in_folder_imgs, out_file):
    pd.options.mode.chained_assignment = None

    # Load csv data and needed variables
    df_poles_adjusted = pd.read_csv(out_file)
    df_poles_adjusted["type"] = df_poles_adjusted["type"].astype("Int64")
    idx = 0

    # Load model
    type_classifier = xgb.XGBClassifier()
    type_classifier.load_model("models/type_classifier.json")

    # Loop over extracted segmentation examples
    while idx < len(df_poles_adjusted):
        if df_poles_adjusted.loc[idx, "code"] == 1:
            obj = df_poles_adjusted.iloc[idx]
            img_number = df_poles_adjusted.loc[idx, 'index']
            obj, esc, code = adjust_pole_statistics(in_folder_imgs, img_number, obj)
            if esc is True:
                return 0
            elif code < 0:
                df_poles_adjusted.loc[idx, "code"] = -2  # adjust label of not street lights
                df_poles_adjusted.to_csv(out_file, index=False)
            else:
                # Assign pole type probs using adjusted pole statistics and update pole in database
                obj = get_pole_type_probs(type_classifier, obj)
                df_poles_adjusted.loc[idx] = obj
                df_poles_adjusted.loc[idx, "code"] = 4  # adjust label of corrected poles
                df_poles_adjusted.to_csv(out_file, index=False)
        idx += 1

    print("There are no validated street lights that need correction.")


# Function to validate pole type
def validate_type(in_folder_imgs, out_file):  # noqa: C901
    pd.options.mode.chained_assignment = None
    cv2.namedWindow("check poles")

    # Load csv data and needed variables
    df_poles_adjusted = pd.read_csv(out_file)
    df_poles_adjusted["type"] = df_poles_adjusted["type"].astype("Int64")
    idx, types_dict = 0, {}

    # Get idxs, preds and probs of all street lights that need a type assignment
    sub_df = df_poles_adjusted.loc[
        (df_poles_adjusted["type"].isna()) & (df_poles_adjusted["code"].isin([0, 4]))
    ]

    # Put idx, preds and probs in usable dictionary grouped by most likely street light type
    for i, obj in sub_df.iterrows():
        preds, probs = ast.literal_eval(obj["type_preds"]), ast.literal_eval(obj["type_probs"])
        if preds[0] not in types_dict:
            types_dict[preds[0]] = [[i, preds, probs]]
        else:
            types_dict[preds[0]].append([i, preds, probs])
    types_dict = collections.OrderedDict(sorted(types_dict.items()))

    # Loop over street lights to predict
    i = 0
    back = False
    while i < len(types_dict.keys()):
        key = list(types_dict.keys())[i]
        if back is False:
            j = 0
        while j < len(types_dict[key]):
            street_light = types_dict[key][j]
            idx, preds = street_light[0], street_light[1]
            img_name = in_folder_imgs + "object_all_axes/" + str(idx) + ".png"

            # Load segmenatation example
            if os.path.exists(img_name):
                img = cv2.imread(img_name)
                h, w, _ = img.shape
                img = cv2.resize(img, (int(w * 0.7), int(h * 0.7)))
                cv2.moveWindow("check poles", 800, 50)
                cv2.setWindowTitle("check poles", f"Check pole {idx}")
                cv2.imshow("check poles", img)

                # Determine type or exit the program
                obj, esc = determine_pole_type(preds, df_poles_adjusted.loc[idx])
                back = False
                if esc is True:
                    cv2.destroyWindow("check poles")
                    return 0
                elif esc == "back":
                    back = True
                    if j == 0:
                        if i > 0:
                            prev_key = list(types_dict.keys())[i - 1]
                            j = max(0, len(types_dict[prev_key]) - 1)
                            i -= 1
                    else:
                        j -= 1
                    break
                else:
                    # Assign pole type probs using adjusted pole statistics
                    # and update pole in database
                    df_poles_adjusted.loc[idx] = obj
                    df_poles_adjusted.to_csv(out_file, index=False)

            if back is False:
                j += 1
        if back is False:
            i += 1

    print("There are no more validated street lights that need type determination.")
    cv2.destroyWindow("check poles")


if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser(description="RandLA-Net and SCFNet implementation.")
    parser.add_argument("--validate_pole", action="store_true", required=False)
    parser.add_argument("--adjust_fit", action="store_true", required=False)
    parser.add_argument("--validate_type", action="store_true", required=False)
    parser.add_argument("--input_file", type=str, default="data/csv_files/poles_extracted_amsterdam_oost_part_1.csv")
    parser.add_argument("--in_folder_images", type=str, default="data/images/objects/")
    args = parser.parse_args()

    if not args.validate_pole and not args.adjust_fit and not args.validate_type:
        print("Select one of the following options: --validate_pole --adjust_fit --validate_type")
        sys.exit(1)

    in_folder_images = args.in_folder_images
    csv_poles = args.input_file
    out_file = csv_poles.replace(".csv", "_checked.csv")

    no_validated_warning = (
        "Make sure that (some) poles are already validated using the --validate_pole argument."
    )

    if args.validate_pole:
        validate_poles(in_folder_images, csv_poles, out_file)

    elif args.adjust_fit:
        try:
            adjust_fit(in_folder_images, out_file)
        except Exception as e:
            print(e)
            print(no_validated_warning)
            sys.exit(1)

    elif args.validate_type:
        try:
            validate_type(in_folder_images, out_file)
        except Exception as e:
            print(e)
            print(no_validated_warning)
            sys.exit(1)
