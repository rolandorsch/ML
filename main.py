import sys
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
TOKENIZER_PATH = 'tokenizer.pkl'
MODEL_PATH = 'model.h5'

# Load the tokenizer from the file
tokenizer_filename = TOKENIZER_PATH
with open(tokenizer_filename, 'rb') as f:
    tokenizer = pickle.load(f)

# Load Tensorflow model
model = keras.models.load_model(MODEL_PATH)
print(model.summary())


if len(sys.argv) > 0:
    print("argyumentos")
    arguments = sys.argv[1:]

    # Process the arguments
    for arg in arguments:
        print(arg)

    sequences = tokenizer.texts_to_sequences(arguments)
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH,
                           padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    print(model.predict(padded))


else:
    print('0 or more than 1 arguments received')
