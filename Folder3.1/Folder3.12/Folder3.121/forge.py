import tensorflow as tf
import keras as k
import numpy as np
from PIL import Image
import os


class Everything:
    def __init__(self, names=['Van Gogh', "Picasso"], dir='data'):
        self.names = names  # The labels
        self.dir = dir  # The directory of the images
        # self.key = key # The image names will be what will label; now all we need is a key

    def resize(self, dir='originalData', newDir='data', num=100):
        for f in os.listdir(dir):
            i = Image.open(dir + '/' + f)
            i = i.resize((num, num))
            i.save(newDir + '/' + f)
        return 0

    def labelled(self, dir='data', names=['Van Gogh', "Picasso"]):
        '''
        This will take in the directory, look at the images, label the images, and return two numpy arrays.
        INPUT: (directory where the image data is stored, label aka the names of the painters)
        OUTPUT: (numpy array of pixels, numpy array of labels)
        '''
        # print(os.listdir(dir))
        # pix = Image.open('data/0.Irises.jpg')
        # print(pix)
        # pix = np.asarray(pix)
        # print(pix)
        labels = []
        pix = []
        self.len = len(os.listdir(dir))
        for pa in os.listdir(dir):
            pix.append(np.array(Image.open(dir + "/" + pa)))
            labels.append((int(pa.split('.')[0]))) # 0 is a forgery; 1 is authentic
        return pix, np.array(labels)

    def buildModel(self):
        m = k.models.Sequential()
        m.add(k.layers.Conv2D(10, kernel_size=(2, 2),
                              activation='relu', input_shape=(100, 100, 3)))
        m.add(k.layers.MaxPooling2D(pool_size=(5, 5)))
        m.add(k.layers.Dropout(0.4))
        m.add(k.layers.Flatten())
        m.add(k.layers.Dense(128, activation="relu"))
        m.add(k.layers.Dropout(0.5))
        m.add(k.layers.Dense(1, activation="sigmoid"))
        m.compile(loss=k.losses.mean_squared_error,
                  optimizer=k.optimizers.Adam(), metrics=['accuracy'])
        return m

    # def initLabels(self, names, labels):
    #     di = {}
    #     for i in range(len(names)):
    #         di[names[i]] = i
    #     newLab = np.array([di[i] for i in labels])
    #     # labels = labels.reshape(len(labels), 1)
    #     newLab = k.utils.to_categorical(newLab, len(names))
    #     return newLab

    def initPix(self, pix, sz=100, num=14):
        pix = np.divide(pix, np.array(255))
        pix.reshape((num, sz, sz, 3))
        return pix

    def trainIt(self, names=np.array(['Van Gogh', "Picasso"]), dir='data', dataNum=14):
        mod = self.buildModel()
        trainPix, trainLab = self.labelled(dir, names)
        # trainLab = self.initLabels(names, trainLab)
        trainLab.reshape(dataNum, 1)
        # trainLab = k.utils.to_categorical(trainLab)
        trainPix = self.initPix(trainPix)
        print(np.shape(trainLab), np.shape(trainPix))
        mod.fit(trainPix, trainLab, batch_size=1, epochs=25, verbose=1)
        self.load = "artForge.hdf5"
        mod.save(self.load)
        self.model = mod
        return mod
    # def testUp(self):


i = Everything(["Van Gogh", "Picasso"], "data")
# i.resize()
# # print(i.labelled("data", ["Van Gogh"]))
i.trainIt()
