import tensorflow as tf 
import keras as k 
import numpy as np
import random

class Name :
    def openAndParse(self, file = 'names.txt'):
        q = open(file)
        i = 0
        names = []
        for line in q:
            if (line[-1] == '\n'): names.append(line[0:-1])
            else: names.append(line)
        self.names = names
        return names
    def findMax(self, arr):
        m = len(arr[0])
        for i in arr:
            if (len(i) > m): m = len(i)
        return m
    def padIt(self, names):
        padNum = self.findMax(names)
        for i in range(len(names)):
            # print(i, len(i))
            while (len(names[i]) !=padNum): names[i] = names[i] + ' '
            # print(names[i], len(names[i]))
        return names
    def buildMod(self, padNum):
        m = k.models.Sequential()
        m.add(k.layers.Dense(1, activation = 'relu', input_shape=(1,)))
        m.add(k.layers.Dense(14, activation = 'relu'))
        m.add(k.layers.Dense(15, activation='relu'))
        m.add(k.layers.Dense(padNum-1, activation='sigmoid'))
        m.compile(loss=k.losses.mean_squared_error, optimizer=k.optimizers.SGD())
        return m

    def makeKey(self):
        # Making a dictionary
        alph = np.array(list(' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))
        di = {}
        for i in range(len(alph)): di[alph[i]] = i
        self.alph = alph
        self.di = di
        return alph, di
    
    def parseDataAndTrain(self):
        alph, di = self.makeKey()
        names = self.padIt(self.openAndParse())
        # for i in names:
        #     print(len(i))
        # print(names)
        # print(names)
        padNum = self.findMax(names)
        mod = self.buildMod(padNum)
        inp = np.array([di[i[0]]/len(alph) for i in names])
        inp.reshape(len(names), 1)
        out = []
        for i in names:
            ne = list(i)[1:]
            for j in range(len(ne)):
                ne[j] = di[ne[j]]
            out.append(ne)
        out = np.array(out)
        out = np.divide(out, np.array(len(alph)))
        out.reshape(len(names), padNum-1)
        # print(inp, out)
        mod.fit(
            inp,
            out,
            batch_size = 1,
            epochs=100,
            verbose=1
        )
        mod.save('names.hdf5')
        return mod
    def generate(self, path = 'names.hdf5'):
        # mod = mod.load(path)
        mod = self.parseDataAndTrain()
        # mod.predict('A')
        # mod.predict('S')
        start = self.alph[random.randint(1, 26)]
        # print(start, self.di)
        go = np.array(self.di[start])
        go.resize(1, )
        # print(alph[round(mod.predict(go))])
        x = mod.predict(go)
        # print(x)
        x = np.multiply(x, np.array(len(self.alph)))
        # print(x)
        ans = start
        x = np.trunc(x)
        # print(x)
        x = np.ndarray.tolist(x)[0]
        # print(x)
        for i in x:
            ans+=self.alph[int(i)]
        # print(ans)
        return ans

i = Name()
# print(i.parseDataAndTrain)
print(i.generate())

# q = open('names.txt')
# i = 0
# names = []
# for line in q:
#     if (line[-1] == '\n'): names.append(line[0:-1])
#     else: names.append(line)

# def findMax(arr):
#     m = len(arr[0])
#     for i in arr:
#         if (len(i) > m): m = len(i)
#     return m
# def padIt(names):
#     padNum = findMax(names)
#     for i in range(len(names)):
#         # print(i, len(i))
#         while (len(names[i]) !=padNum): names[i] = names[i] + ' '
#         # print(names[i], len(names[i]))
#     return names 

# names = padIt(names)
# # for i in names:
# #     print(len(i))
# # print(names)

# def buildMod(padNum):
#     m = k.models.Sequential()
#     m.add(k.layers.Dense(1, activation = 'relu', input_shape=(1,)))
#     m.add(k.layers.Dense(14, activation = 'relu'))
#     m.add(k.layers.Dense(15, activation='relu'))
#     m.add(k.layers.Dense(padNum-1, activation='sigmoid'))
#     m.compile(loss=k.losses.mean_squared_error, optimizer=k.optimizers.SGD())
#     return m


# # Making a dictionary
# alph = np.array(list(' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))
# di = {}
# for i in range(len(alph)): di[alph[i]] = i


# print(names)
# padNum = findMax(names)
# mod = buildMod(padNum)
# inp = np.array([di[i[0]]/len(alph) for i in names])
# inp.reshape(len(names), 1)
# out = []
# for i in names:
#     ne = list(i)[1:]
#     for j in range(len(ne)):
#         ne[j] = di[ne[j]]
#     out.append(ne)
# out = np.array(out)
# out = np.divide(out, np.array(len(alph)))
# out.reshape(len(names), padNum-1)
# print(inp, out)
# mod.fit(
#     inp, 
#     out, 
#     batch_size = 1, 
#     epochs=100, 
#     verbose=1
# )
# # mod.predict('A')
# # mod.predict('S')
# start = 'A'
# go = np.array(di[start])
# go.resize(1, )
# # print(alph[round(mod.predict(go))])
# x = mod.predict(go)
# print(x)
# x = np.multiply(x, np.array(len(alph)))
# print(x)
# ans = start
# x = np.round(x)
# # print(x)
# x = np.ndarray.tolist(x)[0]
# print(x)
# for i in x:
#     ans+=alph[int(i)]
# print(ans)
