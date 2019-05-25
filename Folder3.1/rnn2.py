from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import numpy as np

# data = '''Let me not to the marriage of true minds 
# Admit impediments. Love is not love 
# Which alters when it alteration finds, 
# Or bends with the remover to remove. 
# O no! it is an ever-fixed mark 
# That looks on tempests and is never shaken; 
# It is the star to every wand'ring bark, 
# Whose worth's unknown, although his height be taken. 
# Love's not Time's fool, though rosy lips and cheeks 
# Within his bending sickle's compass come; 
# Love alters not with his brief hours and weeks, 
# But bears it out even to the edge of doom. 
# If this be error and upon me prov'd, 
# I never writ, nor no man ever lov'd.'''
# Hey, sonnet 116!!!!!!! FOR MR. C!

fin = open("shakespeare.txt", "r")
# print(fin.read())
data = fin.read()
print(data)

tok = Tokenizer() # Basically this thing will make tokens out of the words.
# It's boring though... We're using a LIBRARY!!!!!!!!!!!!!!

def parse_data(d):
    d = d.split("\n") # now it says to make it lower case...
    # d = list(d)
        # Why would I do that? It's sort of boring that way...
    tok.fit_on_texts(d) # Badabing, badaboom! You fit it on!
    tot = len(tok.word_index) + 1 # Total # of words
    # print(tok.fit_on_texts(d))
    inp = []
    for i in d:
        li = tok.texts_to_sequences([i])[0] # This will translate it to lists of integers for the text
        # print("FIRST", li)
        for j in range(1, len(li)): # Iterates through each
            stuff = li[:j+1]
            # print(stuff)
            # So what's happening here is that it is starting with two words and adding on another words each time
            # (Note that the words are in the form of vectorized text...)
            inp.append(stuff)

    maxLEN = max(len(x) for x in inp) # Gets the max length
    inp = np.array(pad_sequences(inp, maxlen=maxLEN, padding = 'pre')) # This pads 0s from the start...
    # print(inp)
    pred, lab = inp[:, :-1], inp[:, -1]
    lab = ku.to_categorical(lab, num_classes=tot) # Parse data to its own vector
    # print(pred)
    # print(lab)
    return pred, lab, maxLEN, tot


def model(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    m = Sequential()
    m.add(Embedding(total_words, 10, input_length=input_len))
    m.add(LSTM(150))
    m.add(Dropout(0.1))
    m.add(Dense(total_words, activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer='adam')
    m.fit(predictors, label, epochs=10, verbose=1)
    return m


def generate_text(seed_text, next_words, max_sequence_len, model):
    for j in range(next_words):
        token_list = tok.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=1)

        output_word = ""
        for word, index in tok.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

predict, label, maxLEN, tot = parse_data(data)

model = model(predict, label, maxLEN, tot)

text = generate_text("SARAH:", 100, maxLEN, model)
print(text)
