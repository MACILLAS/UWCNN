# This code generates synthetic underwater images
# Future: synthetic underwater images with synthetic artificial illumination
# Code Inspired By: Li C, Anwar S, Porikli F. Underwater scene prior inspired deep learning image and video enhancement[J]. Pattern Recognition, 2020, 98: 107038
# Implementation By: Max Midwinter

import cv2
import numpy as np
import random

def randomCircle (m1, m2, numspots=2):
    '''
    this function takes a numpy array mask and returns mask with a circle
    of stocastic dimension, location and intensity

    :param m1: np.ones of desired mask shape (this mask removes circles from augmented images)
    :param m2: np.zeros of desired mask shape (this masks crops same circles from the gt image)
    :return: numpy mask dtype=float32
    '''
    row, col = m1.shape

    for i in range(numspots):
        row = random.randint(0, row)
        col = random.randint(0, col)
        # hard coded size between 50 to 125
        r = random.randint(50, 125)
        cv2.circle(m1, center=(col, row), radius=r, color=0, thickness=-1)
        cv2.circle(m2, center=(col, row), radius=r, color=1, thickness=-1)
    return m1, m2


def main(light=True, numsets=2):
    datadir = './NYU_GT/'
    outdir = './NYU_UW_type9/'

    # beta-coefficients of degradation by turbidity
    type_I = [0.85, 0.961, 0.982]
    type_IA = [0.84, 0.955, 0.975]
    type_IB = [0.83, 0.95, 0.968]
    type_II = [0.80, 0.925, 0.94]
    type_III = [0.75, 0.885, 0.89]
    type_1 = [0.75, 0.885, 0.875]
    type_3 = [0.71, 0.82, 0.8]
    type_5 = [0.67, 0.73, 0.67]
    type_7 = [0.62, 0.61, 0.5]
    type_9 = [0.55, 0.46, 0.29]

    # set turbidity type
    TYPE = np.array(type_9)

    # augment num_times
    for b in range(1, numsets+1):
        num_batch = str(b)
        # filenames in NYU_GT are X_Depth_.bmp and X_Image_.bmp where X [1:1449]
        for i in range(1, 1449):
            imgIndex = str(i)

            # read image, convert to ndarray, normalize [0-1]
            image = cv2.imread(datadir + imgIndex + "_Image_.bmp")
            image = np.array(image)
            image0 = image[10:-10, 10:-10, :] / 255

            # read depth, convert to ndarray, normalize [0-1]
            depth = cv2.imread(datadir + imgIndex + "_Depth_.bmp", -1)
            depth = np.array(depth)
            depth0 = depth[10:-10, 10:-10] / 255

            deep = 5 - 2 * random.uniform(0, 1)
            horization = 15 - 15.4 * random.uniform(0, 1)

            # How many artifical images to generate from one sample...
            # Number = 10
            # BGR --> RGB
            A = 1.5 * np.power(TYPE, deep)
            row, col = depth0.shape
            t = np.empty([row, col, 3])
            t[:, :, 0] = np.power(TYPE[0], depth0 * horization)
            t[:, :, 1] = np.power(TYPE[1], depth0 * horization)
            t[:, :, 2] = np.power(TYPE[2], depth0 * horization)

            # Generate the Underwater Images

            I = np.empty([row, col, 3])
            I = A * image0 * t + (1 - t) * A
            I = I * 255
            I = I.astype('float32')

            if light:
                # Now we are going to add in the artificial illumination
                # V3. Instead of changing the saturation after augmentation when true colour is lost
                # We are going to use the gt image (cuz from observation artificially illuminated images look like gt)

                # change image to float32 format
                image0 = image0.astype('float32')

                # sat modification
                image0_sat = cv2.cvtColor(image0, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(image0_sat)
                s = np.clip(s, 0, 0.4)
                # clip the saturation values
                image0_sat = cv2.merge((h, s, v))
                image0_sat = cv2.cvtColor(image0_sat, cv2.COLOR_HSV2RGB)

                # section removal mask: m1 (ones, maskval: 0)
                # image crop mask: m2 (zeros, maskval: 1)
                m1 = np.ones(image0.shape[:2], dtype="float32")
                m2 = np.zeros(image0.shape[:2], dtype="float32")
                m1, m2 = randomCircle(m1, m2)

                row, col, chn = image0.shape

                # the augmented image with black circles
                I_cropped = I * np.dstack((m1, m1, m1))
                # the gt image black except for the circles at regular colour
                image0_cropped = image0_sat * np.dstack((m2, m2, m2))

                comp_image = I_cropped + (image0_cropped*255)

                #maskchn = np.ones((row, col, chn), dtype="float32")
                #maskchn[:, :, 1] = mask
                #hsv = hsv * maskchn
                #image0 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                comp_image = cv2.cvtColor(comp_image, cv2.COLOR_RGB2BGR)
                print(str(outdir+imgIndex+"_UWImage_.bmp"))
                cv2.imwrite(str(outdir+imgIndex+"_UWImage_" + num_batch + "_.bmp"), comp_image)

            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            print(str(outdir + imgIndex + "_UWImage_NL_" + num_batch + "_.bmp"))
            cv2.imwrite(str(outdir + imgIndex + "_UWImage_NL_" + num_batch + "_.bmp"), I)

if __name__ == "__main__":
    main(light=True, numsets=3)
