import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, SpatialDropout2D, ReLU, Input, Concatenate, Add
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import cv2

class UWCNN(tf.keras.Model):

    def __init__(self):
        super(UWCNN, self).__init__()
        self.conv1 = Conv2D(16, 3, (1, 1), 'same', name="conv2d_dehaze1")
        self.relu1 = ReLU()
        self.conv2 = Conv2D(16, 3, (1, 1), 'same', name="conv2d_dehaze2")
        self.relu2 = ReLU()
        self.conv3 = Conv2D(16, 3, (1, 1), 'same', name="conv2d_dehaze3")
        self.relu3 = ReLU()
        self.concat1 = Concatenate(axis=3)

        self.conv4 = Conv2D(16, 3, (1, 1), 'same', name="conv2d_dehaze4")
        self.relu4 = ReLU()
        self.conv5 = Conv2D(16, 3, (1, 1), 'same', name="conv2d_dehaze5")
        self.relu5 = ReLU()
        self.conv6 = Conv2D(16, 3, (1, 1), 'same', name="conv2d_dehaze6")
        self.relu6 = ReLU()
        self.concat2 = Concatenate(axis=3)

        self.conv7 = Conv2D(16, 3, (1, 1), 'same', name="conv2d_dehaze7")
        self.relu7 = ReLU()
        self.conv8 = Conv2D(16, 3, (1, 1), 'same', name="conv2d_dehaze8")
        self.relu8 = ReLU()
        self.conv9 = Conv2D(16, 3, (1, 1), 'same', name="conv2d_dehaze9")
        self.relu9 = ReLU()
        self.concat3 = Concatenate(axis=3)

        self.conv10 = Conv2D(3, 3, (1, 1), 'same', name="conv2d_dehaze10")
        self.add1 = Add()

    def call(self, inputs):
        image_conv1 = self.relu1(self.conv1(inputs))
        image_conv2 = self.relu2(self.conv2(image_conv1))
        image_conv3 = self.relu3(self.conv3(image_conv2))
        dehaze_concat1 = self.concat1([image_conv1, image_conv2, image_conv3, inputs])

        image_conv4 = self.relu4(self.conv4(dehaze_concat1))
        image_conv5 = self.relu5(self.conv5(image_conv4))
        image_conv6 = self.relu6(self.conv6(image_conv5))
        dehaze_concat2 = self.concat2([dehaze_concat1, image_conv4, image_conv5, image_conv6])

        image_conv7 = self.relu7(self.conv7(dehaze_concat2))
        image_conv8 = self.relu8(self.conv8(image_conv7))
        image_conv9 = self.relu9(self.conv9(image_conv8))
        dehaze_concat3 = self.concat3([dehaze_concat2, image_conv7, image_conv8, image_conv9])

        image_conv10 = self.conv10(dehaze_concat3)
        out = self.add1([inputs, image_conv10])
        return out

def parse_function(filename, label):
    filename_image_string = tf.io.read_file(filename)
    label_image_string = tf.io.read_file(label)
    # Decode the filename_image_string
    filename_image = tf.image.decode_bmp(filename_image_string, channels=3)
    filename_image = tf.image.convert_image_dtype(filename_image, tf.float32)
    # Decode the label_image_string
    label_image = tf.image.decode_bmp(label_image_string, channels=3)
    label_image = tf.image.convert_image_dtype(label_image, tf.float32)
    return filename_image, label_image

def combloss (y_actual, y_predicted):
    '''
    This is the custom loss function for keras model
    :param y_actual:
    :param y_predicted:
    :return:
    '''
    # this is just l2 + lssim
    lssim = tf.constant(1, dtype=tf.float32) - tf.reduce_mean(tf.image.ssim(y_actual, y_predicted, max_val=1, filter_size=13)) #remove max_val=1.0
    lmse = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(y_actual, y_predicted)
    lmse = tf.math.multiply(lmse, 4)
    return tf.math.add(lmse, lssim)

def train(datafile="data.csv", ckptpath="./train_type1/cp.ckpt", type='type1'):
    df = pd.read_csv(datafile)
    augfiles = list(df["AUGFILE"])
    gtfiles = list(df["GTFILE"])

    augImages = tf.constant(augfiles)
    gtImages = tf.constant(gtfiles)

    dataset = tf.data.Dataset.from_tensor_slices((augImages, gtImages))
    dataset = dataset.shuffle(len(augImages))
    #dataset = dataset.repeat()
    dataset = dataset.map(parse_function).batch(10)

    # Call backs
    #checkpoint_path = "./train_type1/cp.ckpt"
    checkpoint_path = ckptpath
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model = UWCNN()
    model.compile(optimizer=Adam(), loss=combloss)
    model.fit(dataset, epochs=40, callbacks=[cp_callback])

    os.listdir(checkpoint_dir)
    #model.save('saved_model/my_model')
    model.save('save_model/'+type)

def model_test(imgdir="./test_images/", imgfile="12433.png", ckdir="./train_type1/cp.ckpt", outdir="./results/", type='type1'):
    model = tf.keras.models.load_model('save_model/'+type, custom_objects={'loss': combloss}, compile=False)
    model.summary()
    model.compile(optimizer=Adam(), loss=combloss)
    model.load_weights(ckdir)
    filename_image_string = tf.io.read_file(imgdir+imgfile)
    filename_image = tf.image.decode_png(filename_image_string, channels=3)
    filename_image = tf.image.convert_image_dtype(filename_image, tf.float32)
    filename_image = tf.image.resize(filename_image, (460, 620))
    l, w, c = filename_image.shape
    filename_image = tf.reshape(filename_image, [1, l, w, c])
    output = model.predict(filename_image)
    output = output.reshape((l, w, c)) * 255
    cv2.imwrite(outdir+type+"_"+imgfile, output)

if __name__ == "__main__":
    train(datafile="data_type1.csv", ckptpath="./train_type1/cp.ckpt", type='type1')

    # type = "type1"
    # ckdir = "./train_type1/cp.ckpt"
    # model_test(imgdir="./test_images/", imgfile="532_img_.png", ckdir=ckdir, outdir="./results/", type=type)
    # model_test(imgdir="./test_images/", imgfile="602_img_.png", ckdir=ckdir, outdir="./results/", type=type)
    # model_test(imgdir="./test_images/", imgfile="617_img_.png", ckdir=ckdir, outdir="./results/", type=type)
    # model_test(imgdir="./test_images/", imgfile="12422.png", ckdir=ckdir, outdir="./results/", type=type)
    # model_test(imgdir="./test_images/", imgfile="12433.png", ckdir=ckdir, outdir="./results/", type=type)
