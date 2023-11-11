from EEGModels import EEGNet
import tensorflow as tf
import numpy as np
import pickle as p
import pandas as pd

with open('all_users.pkl', 'rb') as f:
    data = p.load(f)
users = list(data.keys())
accuracies = np.ones((len(users), len(users)), dtype=float)
for i in range(len(users)):
    user_train = users[i]
    for j in range(len(users)):
        if i != j:
            user_test = users[j]
            [user_train_x, user_train_y] = data[user_train]
            user_train_x = user_train_x.reshape(user_train_x.shape[0], 56, 260, 1)
            [user_test_x, user_test_y] = data[user_test]
            user_test_x = user_test_x.reshape(user_test_x.shape[0], 56, 260, 1)
            model = EEGNet(nb_classes=1, Chans=56, Samples=260)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/checkpoint1.h5', verbose=1, save_best_only=True)
            vc = np.unique(user_train_y, return_counts=True)[1]
            cw = {0:(vc[1] / vc[0]), 1:1}
            model.fit(user_train_x, user_train_y, batch_size=16, epochs=300, verbose=2, callbacks=[checkpointer], class_weight=cw)
            probabilities = model.predict(user_test_x)
            predictions = (probabilities.flatten() >= 0.5).astype(int)
            accuracies[i, j] = np.mean(predictions == user_test_y)
DF = pd.DataFrame(accuracies)
DF.to_csv('baseline_accuracies.csv')