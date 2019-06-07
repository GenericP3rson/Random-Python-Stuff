import tensorflow as tf 
import keras as k
import numpy as np 

class Name: 
    def __init__(self):
        self.char = ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890\n'
        self.di = {let : num for num, let in enumerate(list(self.char))}
        self.max = len(self.char)
        self.min = 0
    def normalise(self, x):
        '''
        (x - min(x)) / (max(x) - min(x))
        '''
        
    def parseData(self, path='na.txt'):
        data = open(path)
        # Okay, so now we'll need to chop up the data...
        selen = 2 # A semi-arbitrary number, sort of like neurons in a NN
        fullTxt = ''.join([j for r in data for j in r])
        # Now that we have the text, let's normalise it (which is apparently better than standardisation in this case...)

        print(fullTxt)
        numSeq = len(fullTxt) // selen
        print(numSeq)
    def buildModel(self):
        m = k.models.Sequential()

        
i = Name()
i.parseData()