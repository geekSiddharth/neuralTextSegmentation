# % env KERAS_BACKEND = theano
# % env
# THEANO_FLAGS = 'exception_verbosity=high'
import cPickle
from AttentionWithContext import AttentionWithContext
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, GlobalMaxPooling1D, LSTM, \
    Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

with open('train_X', 'rb') as f:
    train_X = cPickle.load(f)

with open('train_Y', 'rb') as f:
    train_Y = cPickle.load(f)

with open('embedding_W', 'rb') as f:
    embedding_W = cPickle.load(f)

ONE_SIDE_CONTEXT_SIZE = 5


def lstm_model(sequences_length_for_training, embedding_dim, embedding_matrix, vocab_size):
    GLOVE_EMBEDDING_DIM = 300

    print
    'Build MAIN model...'
    ngram_filters = [2, 3, 4, 5]
    conv_hidden_units = [200, 200, 200, 200]

    left_context = Input(shape=(ONE_SIDE_CONTEXT_SIZE + 1, embedding_dim), dtype='float32', name='left-context')
    main_input = Input(shape=(1, embedding_dim), dtype='float32', name='main-input')
    right_context = Input(shape=(ONE_SIDE_CONTEXT_SIZE + 1, embedding_dim), dtype='float32', name='right-context')

    context_embedder = TimeDistributed(
        Embedding(vocab_size + 1, GLOVE_EMBEDDING_DIM, input_length=embedding_dim, weights=[embedding_matrix],
                  init='uniform', trainable=False))
    main_input_embedder = TimeDistributed(
        Embedding(vocab_size + 1, GLOVE_EMBEDDING_DIM, input_length=embedding_dim, weights=[embedding_matrix],
                  init='uniform', trainable=False))

    embedded_input_left, embedded_input_main, embedded_input_right = context_embedder(
        left_context), main_input_embedder(main_input), context_embedder(right_context)

    convsL, convsM, convsR = [], [], []
    for n_gram, hidden_units in zip(ngram_filters, conv_hidden_units):
        conv_layer = Convolution1D(nb_filter=hidden_units,
                                   filter_length=n_gram,
                                   border_mode='same',
                                   # border_mode='valid',
                                   activation='tanh', name='Convolution-' + str(n_gram) + "gram")
        lef = TimeDistributed(conv_layer, name='TD-convolution-left-' + str(n_gram) + "gram")(embedded_input_left)
        mid = TimeDistributed(conv_layer, name='TD-convolution-mid-' + str(n_gram) + "gram")(embedded_input_main)
        rig = TimeDistributed(conv_layer, name='TD-convolution-right-' + str(n_gram) + "gram")(embedded_input_right)

        # Use GlobalMaxPooling1D() instead of Flatten()
        pool_L = TimeDistributed(GlobalMaxPooling1D(), name='TD-GlobalMaxPooling-left-' + str(n_gram) + "gram")(lef)
        pool_M = TimeDistributed(GlobalMaxPooling1D(), name='TD-GlobalMaxPooling-mid-' + str(n_gram) + "gram")(mid)
        pool_R = TimeDistributed(GlobalMaxPooling1D(), name='TD-GlobalMaxPooling-right-' + str(n_gram) + "gram")(rig)
        convsL.append(pool_L), convsM.append(pool_M), convsR.append(pool_R)

    convoluted_left, convoluted_mid, convoluted_right = Merge(mode='concat')(convsL), Merge(mode='concat')(
        convsM), Merge(mode='concat')(convsR)
    CONV_DIM = sum(conv_hidden_units)

    flat_mid = Flatten()(convoluted_mid)

    encode_mid = Dense(300, name='dense-intermediate-mid-encoder')(flat_mid)

    context_encoder_intermediate1 = Bidirectional(
        LSTM(600, input_shape=(ONE_SIDE_CONTEXT_SIZE, CONV_DIM), dropout_W=0.3, dropout_U=0.3,
             return_sequences=True, stateful=False), name='BiLSTM-context-encoder-intermediate1', merge_mode='concat')
    context_encoder = Bidirectional(
        LSTM(600, input_shape=(ONE_SIDE_CONTEXT_SIZE, CONV_DIM), dropout_W=0.3, dropout_U=0.3,
             return_sequences=True, stateful=False), name='BiLSTM-context-encoder', merge_mode='concat')

    encode_left = AttentionWithContext()(
        context_encoder(context_encoder_intermediate1(convoluted_left)))
    encode_right = AttentionWithContext()(
        context_encoder(context_encoder_intermediate1(convoluted_right)))

    encode_left_drop, encode_mid_drop, encode_right_drop = Dropout(0.3)(encode_left), Dropout(0.2)(encode_mid), Dropout(
        0.3)(encode_right)

    encoded_info = Merge(mode='concat', name='encode_info')([encode_left_drop, encode_mid_drop, encode_right_drop])

    decoded = Dense(500, name='decoded')(encoded_info)
    decoded_drop = Dropout(0.3, name='decoded_drop')(decoded)

    output = Dense(1, activation='sigmoid')(decoded_drop)
    model = Model(input=[left_context, main_input, right_context], output=output)
    model.layers[1].trainable = False
    # model.compile(loss=w_binary_crossentropy, optimizer='rmsprop', metrics=['accuracy', 'recall'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'recall'])

    print
    model.summary()
    return model


weights = get_class_weights(train_Y)

model = lstm_model(-1, 20, embedding_W, 65700)

checkpoints = ModelCheckpoint('trained_model.{epoch:02d}-{val_loss:.3f}.hdf5',
                              monitor='acc',
                              verbose=1,
                              save_best_only=True,
                              save_weights_only=True,
                              mode='max',
                              period=1)

print
"Going to Train"
# x = [train_X[0][0:100],train_X[1][0:100],train_X[2][0:100]]
model.fit(train_X, train_Y,
          batch_size=500,
          nb_epoch=30,
          class_weight=weights,
          validation_split=0.2,
          shuffle=True,
          verbose=1,
          callbacks=[checkpoints])
