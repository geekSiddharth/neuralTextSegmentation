import _pickle as cPickle
import sys

# import to keep it *
from keras.callbacks import ModelCheckpoint
from model import *

sys.path.append("./data/")
from process_documets import load_dataset
from utills import *

LOAD_DATA_FROM_SCRATCH = True
print("Making model")
model = get_model()
print("Done Making model")

if LOAD_DATA_FROM_SCRATCH:
    X, Y = load_dataset()
    X, Y = prepare_loaded_dataset_for_training(X, Y, ONE_SIDE_CONTEXT_SIZE)

    with open("X.pkl", "wb", encoding='utf-8') as f:
        cPickle.dump(X, f)
    with open("Y.pkl", "wb", encoding='utf-8') as f:
        cPickle.dump(Y, f)
else:
    with open("X.pkl", "r", encoding='utf-8') as f:
        X = cPickle.load(f)
    with open("Y.pkl", "r", encoding='utf-8') as f:
        Y = cPickle.load(f)

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'sparse_categorical_accuracy'])

checkpoints = ModelCheckpoint('trained_model.{epoch:02d}-{val_loss:.3f}.hdf5',
                              monitor='acc',
                              verbose=1,
                              save_best_only=True,
                              save_weights_only=True,
                              mode='max',
                              period=1)

print("Going to Train")

# n = 20
# X = [X[0][:n], X[1][:n], X[2][:n]]
# Y = Y[:n]
model.fit(X, Y,
          batch_size=5,
          epochs=30,
          class_weight=get_class_weights(Y),
          validation_split=0.1,
          shuffle=True,
          verbose=1,
          callbacks=[checkpoints])
