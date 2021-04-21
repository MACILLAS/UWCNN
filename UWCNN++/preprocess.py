import cv2
import numpy as np
import pandas as pd

def resizeNYUGT (indir = "./NYU_GT/", outdir = "./NYU_UW_GT/"):
    for i in range(1, 1449):
        imgIndex = str(i)

        # read image, convert to ndarray, normalize [0-1]
        image = cv2.imread(indir + imgIndex + "_Image_.bmp", -1)
        image = np.array(image)
        image0 = image[10:-10, 10:-10, :]
        image0 = image0.astype('float32')

        print(str(outdir + imgIndex + "_Image_.bmp"))
        cv2.imwrite(str(outdir + imgIndex + "_Image_.bmp"), image0)

def createDataCSV (augdir = "./NYU_UW_type1/", gtdir = "./NYU_UW_GT/", outfile="data_typeIII_NL.csv"):
    df = pd.DataFrame(columns=['AUGFILE', 'GTFILE'])
    for i in range (1, 1449):
        indexSTR = str(i)
        #df = df.append({'AUGFILE': str(augdir+indexSTR+"_UWImage_1_.bmp"), 'GTFILE' : str(gtdir+indexSTR+"_Image_.bmp")}, ignore_index=True)
        df = df.append({'AUGFILE': str(augdir+indexSTR+"_UWImage_NL_1_.bmp"), 'GTFILE' : str(gtdir+indexSTR+"_Image_.bmp")}, ignore_index=True)
        #df = df.append(
        #    {'AUGFILE': str(augdir + indexSTR + "_UWImage_2_.bmp"), 'GTFILE': str(gtdir + indexSTR + "_Image_.bmp")},
        #    ignore_index=True)
        df = df.append(
            {'AUGFILE': str(augdir + indexSTR + "_UWImage_NL_2_.bmp"), 'GTFILE': str(gtdir + indexSTR + "_Image_.bmp")},
            ignore_index=True)
        #df = df.append(
        #    {'AUGFILE': str(augdir + indexSTR + "_UWImage_3_.bmp"), 'GTFILE': str(gtdir + indexSTR + "_Image_.bmp")},
        #    ignore_index=True)
        df = df.append(
            {'AUGFILE': str(augdir + indexSTR + "_UWImage_NL_3_.bmp"), 'GTFILE': str(gtdir + indexSTR + "_Image_.bmp")},
            ignore_index=True)

    df.to_csv(outfile, index=False)

if __name__ == "__main__":
    createDataCSV(augdir="./NYU_UW_typeIII/", outfile="data_typeIII_NL.csv")