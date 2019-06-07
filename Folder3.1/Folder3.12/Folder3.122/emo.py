import numpy as np 
import tensorflow as tf 
import keras as k 

class Emo: 
    def __init__(self):
        self.characters = list(' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*();:"\'\\\n\r\t,.<>')
        self.di = {}
        # for (i in range(len(self.characters))): pass
        for x in range(len(self.characters)): self.di[self.characters[x]] = (x+1)/len(self.characters)
            # self.di[self.characters[i]] = (i+1) / len(self.characters)
    def parseData(self, path = 'emo.txt'):
        fi = open(path)
        da = []
        ans = []
        for i in fi:
            st = i.split()
            ans.append(st[0])
            da.append(np.array([self.di[i] for i in list(' '.join(st[1:]))]))
        da = np.array(da)
        print(da)
        print(ans)

i = Emo()
i.parseData()