# # import tensorflow as tf 
# # import keras as k 
# # import numpy as np 
# import re
# import requests as req 


# # r = req.get(url="http://lookup-service-prod.mlb.com/json/named.search_player_all.bam")
# # r = req.get("http://lookup-service-prod.mlb.com/json/named.search_player_all.bam?sport_code='mlb'&active_sw='Y'&name_part='young%'&search_player_all.col_in=player_id")
# # r = req.get("http://lookup-service-prod.mlb.com/json/named.sport_hitting_tm.bam?league_list_id='mlb'&game_type=0&season=2014&player_id=592789")

# # print(r.content)
# # data = "<p>Hello there!</p>"
# # data = re.sub(r'<.*?>', '', data)
# # print(data)

# r = req.get()

import numpy as np
import tensorflow as tf
import keras as k 

class Hurt:
    def __init__(self):
        self.pos = ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']
        self.c = 60.0
        self.di = {}
        for i in range(len(self.pos)): self.di[self.pos[i]] = (i + 1)/7 # Slightly normalises the data, allowing for wiggle room...
    def openData(self, path = 'raBaDa.txt'):
        da = open(path)
        # Um... so the first is the position, next x is their FF scores for the week, then whether they got hurt or not
        li = []
        ans = []
        for x in da:
            hold = []
            q = x.split()
            hold.append(self.di[q[0]])
            for i in range(len(q)-2):
                hold.append(float(q[i+1])/self.c)
            ans.append((int(q[-1])))
            li.append(np.array(hold))
        li = np.array(li).reshape(len(ans), len(li[0]),)
        ans = np.array(ans).reshape(len(ans),1,)
        print(li)
        print(ans)
        return li, ans
    def buildModel(self, inLen):
        m = k.models.Sequential()
        m.add(k.layers.Dense(10, activation='relu', input_shape=(inLen,)))
        m.add(k.layers.Dropout(0.1))
        m.add(k.layers.Dense(34, activation='relu'))
        m.add(k.layers.Dropout(0.5))
        m.add(k.layers.Dense(10, activation='sigmoid'))
        # m.add(k.layers.Dropout(0.1))
        m.add(k.layers.Dense(1))
        m.compile(optimizer=k.optimizers.Adam(), loss = k.losses.mean_squared_error)
        return m
    def train(self):
        x, y = self.openData()
        mod = self.buildModel(x.shape[1])
        mod.fit(x, y, batch_size=1,epochs=100, verbose=1)
        mod.save('hurt.hdf5')
        # print(mod.predict(self.parseInp(
        #     ['RB', 24.5, 3.45, 34.54, 2.45, 56.45])))
    def parseInp(self, inp):
        pa = [self.di[inp[0]]]
        for i in range(len(inp)-1):
            pa.append(inp[i+1]/self.c)
        pa = np.array(pa)
        pa = pa.reshape(1, len(inp),)
        return pa
    def test(self):
        inp = ['RB', 24.5, 3.45, 34.54, 29.45, 56.45]
        mod = self.buildModel(len(inp))
        inp = self.parseInp(inp)
        mod.load_weights('hurt.hdf5')
        print(inp.shape)
        # print(inp)
        print(mod.predict(inp))
    
        

    

i = Hurt()
i.test()
