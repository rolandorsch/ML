import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
TOKENIZER_PATH = 'tokenizer.pkl'
MODEL_PATH = 'model.h5'

# Load the tokenizer from the file

# tokenizer_filename = TOKENIZER_PATH
# with open(tokenizer_filename, 'rb') as f:
#     tokenizer = pickle.load(f)

# Load Tensorflow model
model = keras.models.load_model(MODEL_PATH)
print(model.summary())


if len(sys.argv) > 0:
    arguments = sys.argv[1:]
    print("argyumentos", arguments)
    # Process the arguments
    for arg in arguments:
        print("argumneto_", arg)

    # sequences = Tokenizer.texts_to_sequences(arguments)
    # padded = pad_sequences(sequences, maxlen=MAX_LENGTH,
    #                       padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    listarg = arguments[0].split(',')
    # print(listarg)
    # x = tf.linspace(0.0, 250, 251)
    # print("a", float(listarg[0]))
    # print("b", int(listarg[1]))
    # print("c", int(listarg[2]))

    x = tf.linspace(float(listarg[0]), int(listarg[1]), int(listarg[2]))

    print(model.predict(x))


else:
    print('0 or more than 1 arguments received')
