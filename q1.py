import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, RNN, LSTM, GRU, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import os
import string
import random

# ------------------ Parameters ------------------
embedding_dim = 64
hidden_units = 128
cell_type = 'LSTM'  
num_samples = 10000  

# ------------------ Load Data ------------------
def load_data(filepath, num_samples):
    input_texts = []
    target_texts = []
    with open(filepath, encoding='utf-8') as f:
        for line in f.readlines()[:num_samples]:
            if '\t' in line:
                input_text, target_text = line.strip().split('\t')[:2]
                target_text = '\t' + target_text + '\n'  # Add start/end tokens
                input_texts.append(input_text)
                target_texts.append(target_text)
    return input_texts, target_texts

# ------------------ Preprocess ------------------
def tokenize(chars):
    vocab = sorted(set(''.join(chars)))
    char2idx = {c: i+1 for i, c in enumerate(vocab)}  # 0 = padding
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

def vectorize(texts, char2idx, maxlen):
    seqs = [[char2idx.get(c, 0) for c in text] for text in texts]
    return pad_sequences(seqs, maxlen=maxlen, padding='post')

# ------------------ Build Model ------------------
def get_rnn_cell(cell_type):
    if cell_type == 'LSTM':
        return LSTM
    elif cell_type == 'GRU':
        return GRU
    else:
        return SimpleRNN

def build_model(input_vocab_size, target_vocab_size, embedding_dim, hidden_units, cell_type):
    cell = get_rnn_cell(cell_type)

    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_outputs, state_h, state_c = cell(hidden_units, return_state=True)(enc_emb)
    encoder_states = [state_h, state_c] if cell_type == 'LSTM' else [state_h]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    dec_emb = Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
    if cell_type == 'LSTM':
        decoder_lstm = cell(hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    else:
        decoder_rnn = cell(hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_rnn(dec_emb, initial_state=encoder_states)

    decoder_dense = Dense(target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# ------------------ Main ------------------
input_texts, target_texts = load_data('path/to/hi.translit.sample.tsv', num_samples)

input_char2idx, input_idx2char = tokenize(input_texts)
target_char2idx, target_idx2char = tokenize(target_texts)

encoder_input_data = vectorize(input_texts, input_char2idx, maxlen=20)
decoder_input_data = vectorize(target_texts, target_char2idx, maxlen=20)

decoder_target_data = np.zeros((len(input_texts), 20, len(target_char2idx) + 1), dtype='float32')
for i, target_text in enumerate(target_texts):
    for t, char in enumerate(target_text[1:]):
        if t < 20:
            decoder_target_data[i, t, target_char2idx.get(char, 0)] = 1.0

model = build_model(len(input_char2idx)+1, len(target_char2idx)+1, embedding_dim, hidden_units, cell_type)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# ------------------ Train ------------------
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64, epochs=10, validation_split=0.2)

