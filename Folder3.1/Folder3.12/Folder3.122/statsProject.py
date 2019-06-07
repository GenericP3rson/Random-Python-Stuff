import numpy as np
import math

class Analyse:
    def __init__(self, file = 'sarahSwimStats.txt'):
        self.file = file
        self.k = open(file)
    def convertToSpeed(self):
        # time = []
        # yard = []
        ans = []
        for x in self.k:
            r = x.split()
            # time.append(float(r[3]))
            # yard.append(int(r[1]))
            ans.append(float(r[3])/int(r[1]))
        # print(ans)
        return ans
        # # print(np.array(time), np.array(yard))
        # speeds = np.divide(np.array(yard), np.array(time))
        # # print(list(speeds)[0])
        # # sp = []
        # # for i in range(len(time)): sp.append(float(yard[i] / time[i]))
        # # sp = [s for s in sp]
        # # x, y = speeds
        # n = []
        # for i in speeds:
        #     # if (type(i) == 'float'): n.append(i)
        #     print(i, type(i))
        # # print(n)
        # print((speeds.shape))
        # return speeds
    def sortedTimes(self):
        justTimes = []
        for x in self.k: justTimes.append(float(x.split()[3]))
        # print(sorted(justTimes))
        return justTimes
    def sortedSpeeds(self):
        speeds = self.convertToSpeed()
        # print(sorted(speeds)[::-1])
        speeds = sorted(list(np.round(np.array(speeds), 2)))
        # print(len(speeds))
        # print((speeds))
        ans = []
        for i in speeds: 
            if (i != 0): 
                print(i)
                ans.append(i)
        print(ans)
        return ans
    def median(self):
        data = self.sortedSpeeds()
        print(len(data))
        med = 0
        if (len(data) % 2 == 1):
            med = data[len(data)/2 - 0.5]
        else:
            med = (data[int(len(data)/2)] + data[int((len(data)/2)-1)])/2
        print(med)
        return med
    def yBar(self):
        speeds = self.sortedSpeeds()
        # print(sum(speeds)/len(speeds))
        return sum(speeds)/len(speeds)
    def stanDev(self):
        yBar = self.yBar()
        tot = 0
        q = self.sortedSpeeds()
        print(self.sortedSpeeds())
        for y in self.sortedSpeeds(): tot = tot + (y-yBar)*(y-yBar)
        # print(tot/(len(self.sortedSpeeds())-1))
        # print(self.sortedSpeeds())
        # print(tot, yBar)

    def hardVariance(self):
        speeds = [0.64, 0.64, 0.67, 0.71, 0.71, 0.72, 0.72, 0.72, 0.74, 0.76, 0.77, 0.79, 0.8, 0.81, 0.82, 0.83,
                  0.83, 0.83, 0.85, 0.85, 0.88, 0.9, 0.9, 0.92, 0.93, 0.94, 0.96, 0.96, 0.98, 0.99, 1.04, 1.05]
        tot = 0
        yBar = self.yBar()
        for y in speeds:
            tot = tot + (y - yBar)*(y - yBar)
        print(tot/(len(speeds)-1))
        return tot/(len(speeds)-1)
    
    def hardStanDev(self):
        speeds = [0.64, 0.64, 0.67, 0.71, 0.71, 0.72, 0.72, 0.72, 0.74, 0.76, 0.77, 0.79, 0.8, 0.81, 0.82, 0.83,
            0.83, 0.83, 0.85, 0.85, 0.88, 0.9, 0.9, 0.92, 0.93, 0.94, 0.96, 0.96, 0.98, 0.99, 1.04, 1.05]
        tot = 0
        yBar = self.yBar()
        for y in speeds:
            tot = tot + (y - yBar)*(y - yBar)
        print(tot/(len(speeds)-1))
        x = tot/(len(speeds)-1)
        x = math.sqrt(x)
        print(x)
        return x

i = Analyse()
i.hardStanDev()
