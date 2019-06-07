import numpy as np 

data = open('nameData.txt', 'r').read() 
'''
import urllib.request
data = urllib.request.urlopen("http://www.chakoteya.net/StarTrek/23.htm").read().decode(encoding='utf-8')
'''
chars = list(set(data))
dataSZ, vocabSZ = len(data), len(chars)
# print('Data has %d, %d unique' % (dataSZ, vocabSZ))

toNum = {let: num for num, let in enumerate(chars)}
toChar = {num: let for num, let in enumerate(chars)}
# Honestly, I might just make it a vector...
# print(toChar)

# vect = np.zeros((vocabSZ))
# vect[toNum['a']] = 1
# print(vect)

hiddenLayers = 100 # How many neurons are in its hidden layer
seqLen = 25 # 25 char generated each time step
rate = 1e-1 # learning rate: how quickly it will abandon old beliefs

wxh = np.random.randn(hiddenLayers, vocabSZ) * 0.01 # Input to hidden
whh = np.random.randn(hiddenLayers, hiddenLayers) * 0.01 # Hidden to hidden
why = np.random.randn(vocabSZ, hiddenLayers) * 0.01 # Hidden to output
bh = np.zeros((hiddenLayers, 1))
by = np.zeros((vocabSZ, 1))

# def feedforward():
#     hs[t] = np.tanh(np.dot(wxh, xs[t])) + np.dot(whh, hs[t-1] + bh)
#     ys[t] = np.dot(why, hs[t])
#     ps[t] = np.exp(ys[t] / np.sum(np.exp(ys[t])))

def lossF(inp, tar, pre): 
    # print(inp, tar, pre)
    xs, hs, ys, ps = {}, {}, {}, {}
    # Input layer, hidden layers, outputs, probabilities
    hs[-1] = np.copy(pre) # Deep copy
    loss = 0

    # Forward pass
    for t in range(len(inp)):
        xs[t] = np.zeros((vocabSZ, 1))
        xs[t][inp[t]] = 1
        hs[t] = np.tanh(np.dot(wxh, xs[t]) + np.dot(whh, hs[t-1]) + bh)
        ys[t] = np.dot(why, hs[t]) + by 
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][tar[t], 0]) # Cross entropy
    # Back prop
    dwxh, dwhh, dwhy = np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why)
    # Mimics the vectors except with zeros
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inp))):
        dy = np.copy(ps[t]) # Deep copy of outputs
        dy[tar[t]] -=1
        dwhy += np.dot(dy, hs[t].T)
        dby += dy 
        dh = np.dot(why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dwxh += np.dot(dhraw, xs[t].T)
        dwhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(whh.T, dhraw)
    for dparam in [dwxh, dwhh, dwhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # Makes sure the numbers don' out of controlt get
    return loss, dwxh, dwhh, dwhy, dbh, dby, hs[len(inp)-1]

def generate(h, see, n):
    x = np.zeros((vocabSZ, 1))
    x[see] = 1
    stuff = []
    # for t in range(n):
    name = 0
    while True:
        h = np.tanh(np.dot(wxh, x) + np.dot(whh, h) + bh)
        y = np.dot(why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        x1 = np.random.choice(range(vocabSZ), p=p.ravel())
        x = np.zeros((vocabSZ, 1))
        x[x1] = 1
        stuff.append(x1)
        if ('\n' == toChar[x1]): 
            if (name == n):
                break
            name += 1
    txt = ''.join(toChar[i] for i in stuff)
    print(txt)

n, p = 0, 0
mwxh, mwhh, mwhy = np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smLoss = -np.log(1.0/vocabSZ)*seqLen # the first ever loss...
while n<= 1000*100:
    if (p+seqLen+1 >= len(data) or n == 0):
        hprev = np.zeros((hiddenLayers, 1))
        p = 0
    inp = [toNum[i] for i in data[p:p+seqLen]]
    tar = [toNum[i] for i in data[p+1:p+seqLen+1]]
    loss, dwxh, dwhh, dwhy, dbh, dby, hprev = lossF(inp, tar, hprev)
    smLoss = smLoss * 0.999 + loss * 0.001

    if (n % 1000 == 0):
        print('BELOW: iter %d, loss: %f' % (n, smLoss) + '\n')
        generate(hprev, np.random.randint(0, len(chars)), 10)
        print('ABOVE: iter %d, loss: %f' % (n, smLoss) + '\n')

    for param, dparam, mem in zip([wxh, whh, why, bh, by], [dwxh, dwhh, dwhy, dbh, dby], [mwxh, mwhh, mwhy, mbh, mby]):
        mem += dparam * dparam
        param += -rate * dparam / np.sqrt(mem + 1e-8)
    p+=seqLen
    n+=1





# print(wxh, whh, why, bh, by)
