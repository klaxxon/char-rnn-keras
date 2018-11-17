import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from timeit import default_timer as timer
from model import build_model, save_weights, load_weights
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

DATA_DIR = './data'
LOG_DIR = './logs'
MODEL_DIR = './model'

BATCH_SIZE = 16
SEQ_LENGTH = 64

model = None
sampmodel = None
vocab_size = 0
char_to_idx = None
idx_to_char = None



class TrainLogger(object):
    def __init__(self, file, resume=0):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = resume
        if not resume:
            with open(self.file, 'w') as f:
                f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)


def read_batches(T, vocab_size):
    length = T.shape[0]
    batch_chars = length // BATCH_SIZE

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH):
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH))
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))
        for batch_idx in range(0, BATCH_SIZE):
            for i in range(0, SEQ_LENGTH):
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i]
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y

def sample(num_chars):
    global sampmodel, model, char_to_idx, vocab_size

    if sampmodel == None:
        sampmodel = Sequential()
        sampmodel.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))
        for i in range(3):
            sampmodel.add(LSTM(256, return_sequences=(i != 2), stateful=True))
            sampmodel.add(Dropout(0.2))

        sampmodel.add(Dense(vocab_size))
        sampmodel.add(Activation('softmax'))
    sampmodel.set_weights(model.get_weights()) 
    sampled = []
    for i in range(num_chars):
        batch = np.zeros((1, 1))
        if sampled:
            batch[0, 0] = sampled[-1]
        else:
            batch[0, 0] = np.random.randint(vocab_size)
        result = sampmodel.predict_on_batch(batch).ravel()
        sample = np.random.choice(list(range(vocab_size)), p=result)
        sampled.append(sample)

    return ''.join(idx_to_char[c] for c in sampled)

def train(text, epochs=100, save_freq=10, resume=False):
    global model, idx_to_char, vocab_size
    if resume:
        print("Attempting to resume last training...")

        model_dir = Path(MODEL_DIR)
        c2ifile = model_dir.joinpath('char_to_idx.json')
        with c2ifile.open('r') as f:
            char_to_idx = json.load(f)

        checkpoints = list(model_dir.glob('weights.*.h5'))
        if not checkpoints:
            raise ValueError("No checkpoints found to resume from")

        resume_epoch = max(int(p.name.split('.')[1]) for p in checkpoints)
        print("Resuming from epoch", resume_epoch)

    else:
        resume_epoch = 0
        char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
        with open(os.path.join(MODEL_DIR, 'char_to_idx.json'), 'w') as f:
            json.dump(char_to_idx, f)

    vocab_size = len(char_to_idx)
    idx_to_char = { i: ch for (ch, i) in list(char_to_idx.items()) }
    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    if resume:
        load_weights(resume_epoch, model)

    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    log = TrainLogger('training_log.csv', resume_epoch)
     
    start = timer()
    for epoch in range(resume_epoch, epochs):
        elapsed = timer() - start
        start = timer()
        print('\nEpoch {}/{}, previous epoch took %0.3fs or %0.3fms/step'.format(epoch + 1, epochs) % (elapsed, 1000.0*elapsed/(T.shape[0]/BATCH_SIZE/SEQ_LENGTH)))
        losses, accs = [], []
        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            loss, acc = model.train_on_batch(X, Y)
            print('Batch {}: loss = {:.4f}, acc = {:.5f}\r'.format(i + 1, loss, acc),end =" ")
            losses.append(loss)
            accs.append(acc)

        log.add_entry(np.average(losses), np.average(accs))
        print()
        print(sample(512))
        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--input', default='nottingham-jigs.txt',
                        help='name of the text file to train from')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10,
                        help='checkpoint save frequency')
    parser.add_argument('--resume', action='store_true',
                        help='resume from previously interrupted training')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(os.path.join(DATA_DIR, args.input), 'r', errors="ignore") as data_file:
        text = data_file.read()
    train(text, args.epochs, args.freq, args.resume)
