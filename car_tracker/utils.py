import cv2
from skimage.transform import resize
import numpy as np 
import pickle

EMPTY = True
NOT_EMPTY = False

#loading the model
model = pickle.load(open("dataset/model/model.p", "rb"))

def is_empty(spot_bgr):
    """
    returns weather a given box is having a car parked or not (EMPTY/NOT_EMPTY)
    Args:
        spot_bgr: input rgb image
    Returs:
        bool: EMPTY/NOT_EMPTY
    """

    flat_data = []

    #resizing the input image
    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    #preforming model prediction
    y_out = model.predict(flat_data)

    if y_out == 0:
        return EMPTY
    else:
        return NOT_EMPTY




def get_parking_spot_boxes(connceted_components):
    """
    Returns the list of bouding box in the connected componenets
    Args:
        connceted_components: connceted_components object generated using opencv and mask image
    Returs:
        slots: list of position vector for each bouding box
    """
    (totalLabesl, label_ids, values, centroid) = connceted_components

    slots= [ ]
    coef = 1

    for i in range(1, totalLabesl):
        #extracting the corrdinates for each bounding box
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

def calc_dif(im1, im2):
    """
    computes the mean difference in pixcels of im1 and im2
    Args:
        im1: first image
        im2: second image
    Returns:
        returns difference in mean value between im1 and im2
    """
    return np.mean(np.mean(im1) - np.mean(im2))