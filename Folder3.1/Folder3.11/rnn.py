# EDITED
from __future__ import absolute_import, division, print_function

import time
import os
import numpy as n
import tensorflow as t
t.enable_eager_execution()

# data = t.keras.utils.get_file(
#     "shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

# So, like, basically, shakespeare.txt is a one-letter recount of Shakespeare...
data = open("shakespeare.txt", 'rb').read().decode(encoding="utf-8")[:250]
# for line in data:
#   print(line)

# This basically figures out the vocab and puts it in a list.
uni = sorted(set(data))
# print(uni)
# print('{} unique chatacters'.format(len(uni)))
cool_obj = {u: i for i, u in enumerate(uni)}  # Obj with "CH": index
# print(cool_obj)
arr = n.array(uni)  # As an array
# print(arr)
num = n.array([cool_obj[i] for i in data])
# print(num)


in_max = 100
epochEx = len(data)
fancyData = t.data.Dataset.from_tensor_slices(
    num)  # Has every single character mapped
# for i in fancyData.take(5):
#   print(arr[i.numpy()])
seq = fancyData.batch(in_max+1, drop_remainder=True)  # In groups
# for i in seq.take(5):
#   print(repr(''.join(arr[i.numpy()])))


def divide_and_conquer(data):
  dataIn = data[:-1]  # All but last
  dataOut = data[1:]  # All but first
  return dataIn, dataOut
# print(divide_and_conquer("SARAH"))


bigData = seq.map(divide_and_conquer)

# for inpu, output in bigData.take(1):
#   print("IN", repr(''.join(arr[inpu.numpy()])))
#   print("OUT", repr(''.join(arr[output.numpy()])))


#   for i, (inp, tar) in enumerate(zip(inpu[:5], output[:5])):
#       print("Step {:4d}".format(i))
#       print("  input: {} ({:s})".format(inp, repr(arr[inp])))
#       print("  expected output: {} ({:s})".format(tar, repr(arr[tar])))

ba = 1
epochSt = epochEx//ba
buff = 10000
bigData = bigData.shuffle(buff).batch(ba, drop_remainder=True)
# bigData


sz = len(uni)
embDim = 256
rnnUni = 1024

if (t.test.is_gpu_available()):
  rNN = t.keras.layers.CuDNNGRU
else:
  import functools as f
  rNN = f.partial(t.keras.layers.GRU, recurrent_activation="sigmoid")


def make_NN(sz=sz, embDim=embDim, rnnUni=rnnUni, ba=ba):
  m = t.keras.Sequential([t.keras.layers.Embedding(sz, embDim, batch_input_shape=[ba, None]),
                          rNN(rnnUni, return_sequences=True,
                              recurrent_initializer="glorot_uniform", stateful=True),
                          t.keras.layers.Dense(sz)
                          ])
  return m


m = make_NN()
m.summary()


def loss(lab, log):
  return t.keras.losses.sparse_categorical_crossentropy(lab, log, from_logits=True)


# for inpu, output in bigData.take(1):
#   print("IN", repr(''.join(arr[inpu.numpy()])))
#   print("OUT", repr(''.join(arr[output.numpy()])))
# #   for i, (inp, tar) in enumerate(zip(inpu[:5], output[:5])):
# #       print("Step {:4d}".format(i))
# #       print("  input: {} ({:s})".format(inp, repr(arr[inp])))
# #       print("  expected output: {} ({:s})".format(tar, repr(arr[tar])))
for inp, tar in bigData.take(1):
  pred = m(inp)
#   print(pred.shape, "# (batch_size, sequence_length, vocab_size)")
#   samp = t.random.categorical(pred[0], num_samples=1)
#   samp = t.squeeze(samp,axis=-1).numpy()
#   samp
#   print("Input: \n", repr("".join(arr[inp[0]])))
#   print()
#   print("Next Char Predictions: \n", repr("".join(arr[samp ])))

  exBL = loss(tar, pred)
  print(exBL.numpy().mean())


m.compile(optimizer=t.train.AdamOptimizer(), loss=loss)

home = "./train_check"
chPre = os.path.join(home, "ckpt_{epoch}")
callB = t.keras.callbacks.ModelCheckpoint(
    filepath=chPre, save_weights_only=True)

hist = m.fit(bigData.repeat(), epochs=1,
             steps_per_epoch=epochSt, callbacks=[callB])


def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [cool_obj[s] for s in start_string]
  input_eval = t.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = t.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = t.multinomial(predictions, num_samples=1)[-1, 0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = t.expand_dims([predicted_id], 0)

      text_generated.append(arr[predicted_id])

  return (start_string + ''.join(text_generated))


print(generate_text(m, start_string=u"SARAH: "))
