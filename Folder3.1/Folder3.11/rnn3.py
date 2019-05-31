from __future__ import absolute_import, division, print_function
import time
import os
import numpy as np

import tensorflow as tf
tf.enable_eager_execution()


'''DATA'''
text = open('shakespeare.txt', 'rb').read().decode(encoding='utf-8')
# print('{} unique characters'.format(len(text))) # Got the DATA!!!!
uni = sorted(set(text)) # So this should extract the characters (the set() part) and then sort it
# print(len(uni))
obj = {char:num for num, char in enumerate(uni)}
# Basically, this will give each unique character a corresponding number; 
# enumerate() returns a dictionary of num: "character", and the for loop makes it "character": num;
# print(vect)
arr = np.array(uni) # This is an array of all the characters; the indicies correspond to the desired character
text_stuff = np.array([obj[char] for char in text])

seqLEN = 50 # Max length of an input
numOfExams = len(text) // seqLEN # How many times the seqLEN will evenly go into the data

charData = tf.data.Dataset.from_tensor_slices(text_stuff)
# for i in charData.take(10):
#     print(arr[i])

seq = charData.batch(seqLEN+1, drop_remainder = True)
# for i in seq.take(10):
#     print(repr(''.join(arr[i.numpy()])))
    # print('\nNEXT\n')

def spli(ch):
    inpu = ch[:-1]
    out = ch[1:]
    return inpu, out
data = seq.map(spli)
# print(data)

sz = 64
steps = numOfExams // sz
buffer = 10000

data = data.shuffle(buffer).batch(sz, drop_remainder=True)

# print(data)

voSZ = len(uni)
print(voSZ)

embDIM = 256

rnnNUM = 1024

if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNGRU
  print("YESSSSSS!!!!")
else:
#   print("NOOOOOOO")
  import functools
  rnn = functools.partial(
      tf.keras.layers.GRU, recurrent_activation='sigmoid')


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
      rnn(rnn_units,
          return_sequences=True,
          recurrent_initializer='glorot_uniform',
          stateful=True),
      tf.keras.layers.Dense(vocab_size)
  ])
  return model


model = build_model(
    vocab_size=len(uni),
    embedding_dim=embDIM,
    rnn_units=rnnNUM,
    batch_size=sz)

for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)
#   print(example_batch_predictions.shape,
        # "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(
    example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

# print(sampled_indices)



# Time to TRAIN IT!!!!!!!!
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


example_batch_loss = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape,
#       " # (batch_size, sequence_length, vocab_size)")
# print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)



ep = 3
# history = model.fit(data.repeat(), epochs=ep, steps_per_epoch = steps, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(voSZ, embDIM, rnnNUM, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()


def generate_text(model, start_string):
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [obj[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

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
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(arr[predicted_id])

  return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"SARAH: "))