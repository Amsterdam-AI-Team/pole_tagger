"""Validation tool for street light annotations"""
# Imports
import argparse
import os
import sys
import cv2
import pandas as pd

def validate_annotations(annotations_file, annotation_label, in_folder_images):

    label_dict = {'True Positive': 0, 'False Positive': 2, 'Unsure': 3, 'Corrected': 4, 'FP Corrected': -2}
    label = label_dict[annotation_label]
    non_validated_df = pd.read_csv(annotations_file)
    df = non_validated_df.copy()
    if args.overwrite_file:
        out_file = annotations_file
    else:
        out_file = annotations_file.replace(".csv", "_validated.csv")
    cv2.namedWindow("validate poles")

    df_by_label = df.loc[df['code'] == label]
    if len(df.loc[df['code'] == label]) == 0:
        print('No annotations with label "{}"'.format(annotation_label))
        return 0
    
    i = 0
    identifiers = df_by_label['identifier'].to_list()
    while i < len(df_by_label):
        identifier = identifiers[i]
        obj = df_by_label[df_by_label['identifier'] == identifier]
        img_number = str(obj.iloc[0]['identifier'])
        print(img_number)
        img_file = in_folder_images + img_number + '.png'
        img = cv2.imread(img_file)
        cv2.moveWindow("validate poles", 250, 75)
        if annotation_label == 'Corrected':
            cv2.setWindowTitle("validate poles", f"Index: {img_number}, Labeled as: {annotation_label}, \
New Height: {obj.height}, New Angle: {obj.angle}")
        else:
            cv2.setWindowTitle("validate poles", f"Index: {img_number}, Labeled as: {annotation_label}")
        cv2.imshow("validate poles", img)


        # The function waitKey waits for a key event infinitely (when delay<=0)
        k = None
        pos_values = [110, 13, 32, 3, 99, 102, 117, 8, 127, 2, 49, 50, 51, 52, 81, 83, 116, 2]
        while k not in pos_values:
            k = cv2.waitKey(0)
            if k in [13, 32, 3, 83]:
                # [enter] or [space] or [>]: continue
                continue
            elif k in [116, 49]: # [t] or [1]: true positive
                df.loc[df['identifier'] == identifier, 'code'] = 0
                continue
            elif k == 102 or k == 50:  # [f] or [2]: false positive
                df.loc[df['identifier'] == identifier, 'code'] = 2
                continue
            elif k == 99 or k == 51:  # [c] or [3]: pole needs correction
                df.loc[df['identifier'] == identifier, 'code'] = 1
            elif k == 117 or k == 52:  # [u] or [4]: unclear/more than one pole
                df.loc[df['identifier'] == identifier, 'code'] = 3
                continue
            elif (
                k in [2, 8, 127, 2, 81] and i >= 0
            ):  # [backspace] or [<]: go back one example
                i -= 2
            elif k == 27:  # [esc] to exit the program
                return 0
            else:
                print("Key not valid...")

        df.to_csv(out_file, index=False)

        i += 1
        if i <= 0:
            i = 0

    print("There are no more poles to validate.")

if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser(description="RandLA-Net and SCFNet implementation.")
    parser.add_argument("--annotation_file", type=str, default="data/csv_files/extracted_poles_checked.csv")
    parser.add_argument("--annotation_label", type=str, default='True Positive')
    parser.add_argument("--in_folder_images", type=str, default="data/images/objects/object_all_axes/")
    parser.add_argument("--overwrite_file", action="store_true", required=False)
    args = parser.parse_args()

    csv_annotated_poles = args.annotation_file
    annotation_label = args.annotation_label
    in_folder_images = args.in_folder_images

    csv_directory_warning = (
        "Make sure that the annotation filename and/or directory is correct."
    )
    image_directory_warning = (
        "Make sure that the image directory is correct."
    )
    label_warning = directory_warning = (
        "Make sure that the annotation label is one of following: {}, {}, {}, {} or {}."
        .format( "'True Positive'", "'False Positive'",  "'Unsure'",  "'Corrected'",  "'FP Corrected'")
    )

    if not os.path.exists(csv_annotated_poles):
        print(csv_directory_warning)
        sys.exit(1)
    if not os.path.exists(in_folder_images):
        print(image_directory_warning)
        sys.exit(1)
    if annotation_label not in ['True Positive', 'False Positive', 'Unsure', 'Corrected', 'FP Corrected']:
        print(label_warning)
        sys.exit(1)
    
    validate_annotations(csv_annotated_poles, annotation_label, in_folder_images)


