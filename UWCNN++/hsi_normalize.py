# This code normalizes the output images in HSI space
# Code Inspired By: Li C, Anwar S, Porikli F. Underwater scene prior inspired deep learning image and video enhancement[J]. Pattern Recognition, 2020, 98: 107038
# Implementation By: Max Midwinter
#HSI and RGB conversion code by: DaiPuWei

import os
import cv2
import numpy as np
import math

def RGB2HSI(rgb_img):
    """
         This is the function to convert RGB color image to HSI image
         :param rgm_img: RGB color image
         :return: HSI image
    """
    #Save the number of rows and columns of the original image
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]
    #Copy the original image
    hsi_img = rgb_img.copy()
    #Channel splitting the image
    B,G,R = cv2.split(rgb_img)
    #R, G, B = cv2.split(rgb_img)
    # Normalize the channel to [0,1]
    [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
    H = np.zeros((row, col))    #Define H channel
    I = (R + G + B) / 3.0       #Calculate I channel
    S = np.zeros((row,col))      #Define S channel
    for i in range(row):
        den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
        thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   #Calculate the included angle
        h = np.zeros(col)               #Define temporary array
        #den>0 and G>=B element h is assigned to thetha
        h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
        #den>0 and G<=B element h is assigned to thetha
        h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
        #den<0 element h is assigned a value of 0
        h[den == 0] = 0
        H[i] = h/(2*np.pi)      #Assign to the H channel after radiating
    #Calculate S channel
    for i in range(row):
        min = []
        #Find the minimum value of each group of RGB values
        for j in range(col):
            arr = [B[i][j],G[i][j],R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        #Calculate S channel
        S[i] = 1 - min*3/(R[i]+B[i]+G[i])
        #I is 0 directly assigned to 0
        S[i][R[i]+B[i]+G[i] == 0] = 0
    #Extend to 255 for easy display, generally H component is between [0,2pi], S and I are between [0,1]
    hsi_img[:,:,0] = H*255
    hsi_img[:,:,1] = S*255
    hsi_img[:,:,2] = I*255
    return hsi_img

def HSI2RGB(hsi_img):
    """
         This is the function to convert HSI image to RGB image
         :param hsi_img: HSI color image
         :return: RGB image
    """
    # Save the number of rows and columns of the original image
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    #Copy the original image
    rgb_img = hsi_img.copy()
    #Channel splitting the image
    H,S,I = cv2.split(hsi_img)
    # Normalize the channel to [0,1]
    [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]
    R,G,B = H,S,I
    for i in range(row):
        h = H[i]*2*np.pi
        #H is greater than or equal to 0 and less than 120 degrees
        a1 = h >=0
        a2 = h < 2*np.pi/3
        a = a1 & a2         #Fancy index of the first case
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i]*(1+S[i]*np.cos(h)/tmp)
        g = 3*I[i]-r-b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        #H is greater than or equal to 120 degrees and less than 240 degrees
        a1 = h >= 2*np.pi/3
        a2 = h < 4*np.pi/3
        a = a1 & a2         #Fancy index of the second case
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        #H is greater than or equal to 240 degrees and less than 360 degrees
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2             #Fancy index of the third case
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1-S[i])
        b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:,:,0] = B*255
    rgb_img[:,:,1] = G*255
    rgb_img[:,:,2] = R*255
    return rgb_img

def transform (dir = None):
    img = cv2.imread(dir)
    hsi_img = RGB2HSI(img)/255
    h, s, i = cv2.split(hsi_img)
    s = cv2.normalize(s, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    i = cv2.normalize(i, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    norm_hsi_img = cv2.merge((h, s, i))
    norm_hsi_img = norm_hsi_img*255
    rgb_img = HSI2RGB(norm_hsi_img)
    cv2.imwrite(dir, rgb_img)

def allImgInDir (path = './results'):
    fname = []
    for root, d_names, f_names in os.walk(path):
        for f in f_names:
            fname.append(os.path.join(root, f))
            print("File: "+str(f))
            transform(os.path.join(root, f))
    print("fname  %s" % fname)

if __name__ ==  "__main__":
    allImgInDir('./results')