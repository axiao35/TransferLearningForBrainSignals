import pickle as p
# import pickle5 as p #in case your pickle version is old

with open('all_users.pkl','rb') as f:
    data = p.load(f)
keys = list(data.keys())

from EEGModels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import itertools as it
import random as rd

samples = 260
channels = 56
kernels = 1

all_combinations_2 = list(it.combinations(range(len(keys)), 2))
five_combinations_2 = rd.sample(all_combinations_2, 5)
all_combinations_3 = list(it.combinations(range(len(keys)), 3))
five_combinations_3 = rd.sample(all_combinations_3, 5)
combinations = five_combinations_2 + five_combinations_3

for i in range(len(combinations)):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for key in range(len(keys)):
        [X, Y] = data[keys[key]]
        X = X.reshape(X.shape[0], channels, samples, kernels)
        if key in combinations[i]:
            X_train.append(X)
            Y_train.append(Y)
        else:
            idx = np.random.permutation(X.shape[0])
            spl, rem = idx[:15], idx[15:]
            X_train.append(X[spl, :, :])
            Y_train.append(Y[spl])
            X_test.append(X[rem, :, :])
            Y_test.append(Y[rem])

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    model = EEGNet(nb_classes = 1, Chans = channels, Samples = samples)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1, save_best_only=True)

    vc = np.unique(Y_train, return_counts=True)[1]
    cw = {0:vc[1]/vc[0], 1:1}

    fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
                            verbose = 2, callbacks=[checkpointer], class_weight = cw)

    probs       = model.predict(X_test)
    preds       = (probs.flatten() >= 0.5).astype(int)
    with open('combinations_accuracies.txt', 'a') as f0:
        f0.write('users ' + str(combinations[i] + ': ' + str(round(np.mean(preds == Y_test) * 100, 2))) + '%\n')