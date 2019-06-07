import tensorflow as tf 
import keras as k 
import numpy as np 

class GenerateStats:
    def parseData(self, file='sarahSwimStats.txt'):
        k = open(file)
        swimStuff = {'Freestyle': 0, 'Breaststroke': 1, 'Medley': 2, 'Backstroke': 3, 'Butterfly': 4}
        arr = []
        ans = []
        for i in k:
            q = i.split()
            # t = [float(q[0])/20, float(q[1])/500, swimStuff[q[2]]/5]
            t = [float(q[1])/500, swimStuff[q[2]]/5]
            # for x in range(len(q)-1):
            #     # if (q[x].replace('.', '', 1).isdigit()): t.append(float(q[x]))
            #     # else: t.append(swimStuff[q[x]])
            arr.append(np.array(t))
            ans.append(np.array(float(q[len(q)-1])/500))
        # arr = np.array(arr).reshape(32, 3)
        arr = np.array(arr).reshape(32, 2)
        ans = np.array(ans).reshape(32, 1)
        return arr, ans
    
    def parseInput(self, q):
        swimStuff = {'Freestyle': 0, 'Breaststroke': 1,
                     'Medley': 2, 'Backstroke': 3, 'Butterfly': 4}
        # t = [float(q[0])/20, float(q[1])/500, swimStuff[q[2]]/5]
        t = [float(q[1])/500, swimStuff[q[2]]/5]
        t=np.array(t).reshape(1, 2)
        # print(t)
        return t

    def model(self):
        m = k.models.Sequential()
        m.add(k.layers.Dense(13, activation = 'relu', input_shape = (2, )))
        # m.add(k.layers.Dropout(0.5))
        m.add(k.layers.Dense(100, activation='relu'))
        m.add(k.layers.Dropout(0.75))
        m.add(k.layers.Dense(12, activation='relu'))
        m.add(k.layers.Dropout(0.1))
        m.add(k.layers.Dense(12, activation='relu'))
        m.add(k.layers.Dropout(0.3))
        m.add(k.layers.Dense(50, activation='relu'))
        m.add(k.layers.Dense(1, activation='sigmoid'))
        m.compile(optimizer=k.optimizers.Adam(), loss=k.losses.mean_squared_error)
        return m
    
    def go(self, file='sarahSwimStats.txt'):
        data, labels = self.parseData()
        mod = self.model()
        mod.fit(x=data, y=labels, batch_size=5, epochs=1000, verbose=1)
        mod.save_weights('sarahSwimmingStats1.hdf5')

    def run(self, i):
        mod = self.model()
        mod.load_weights('sarahSwimmingStats1.hdf5')
        for x in i:
            inp = self.parseInput(x)
            # print(inp.shape)
            ans = mod.predict(inp)
            print(np.multiply(ans, np.array(500)))
    



i = GenerateStats()
# i.parseData()
i.go()
i.run([["14", "50", "Freestyle"], ["14", "50",
                                   "Breaststroke"], ["14", "50", "Butterfly"]])
