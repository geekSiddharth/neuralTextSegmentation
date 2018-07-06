from AttentionWithContext import AttentionWithContext
from embedding import *
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Concatenate, Conv1D, GlobalMaxPooling1D, LSTM, \
    Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

SENTENCE_MAX_LEN = 20

ngram_filters = [2, 3, 4, 5]
conv_hidden_units = [200, 200, 200, 200]

ONE_SIDE_CONTEXT_SIZE = 5


def get_model():
    _, embedding_matrix = load_embeddings()

    # Left context input
    left_context = Input(
        shape=(ONE_SIDE_CONTEXT_SIZE + 1, SENTENCE_MAX_LEN),
        name='left-context')

    # Mid line
    main_input = Input(
        shape=(1, SENTENCE_MAX_LEN),
        name='main-input')

    # Right context input
    right_context = Input(
        shape=(ONE_SIDE_CONTEXT_SIZE + 1, SENTENCE_MAX_LEN),
        name='right-context')

    # Will be used as input to left and right context
    context_embedder = TimeDistributed(
        Embedding(VOCAB_SIZE,
                  EMBEDDING_DIM,
                  input_length=SENTENCE_MAX_LEN,
                  weights=[embedding_matrix],
                  embeddings_initializer="uniform",
                  trainable=False)
    )

    # Only for main input
    # TODO: Make trainable=True maybe?
    main_input_embedder = TimeDistributed(
        Embedding(VOCAB_SIZE,
                  EMBEDDING_DIM,
                  input_length=SENTENCE_MAX_LEN,
                  weights=[embedding_matrix],
                  embeddings_initializer="uniform",
                  trainable=False)
    )

    embedded_input_left = context_embedder(left_context)
    embedded_input_main = main_input_embedder(main_input)
    embedded_input_right = context_embedder(right_context)

    """
     Starting to amke the convolution layers
    """
    convsL, convsM, convsR = [], [], []
    for n_gram, hidden_units in zip(ngram_filters, conv_hidden_units):
        conv_layer = Conv1D(filters=hidden_units,
                            kernel_size=n_gram,
                            padding="same",
                            activation='relu',
                            name='Convolution-' + str(n_gram) + "gram")

        lef = TimeDistributed(conv_layer, name='TD-convolution-left-' + str(n_gram) + "gram")(embedded_input_left)
        mid = TimeDistributed(conv_layer, name='TD-convolution-mid-' + str(n_gram) + "gram")(embedded_input_main)
        rig = TimeDistributed(conv_layer, name='TD-convolution-right-' + str(n_gram) + "gram")(embedded_input_right)

        pool_L = TimeDistributed(GlobalMaxPooling1D(), name='TD-GlobalMaxPooling-left-' + str(n_gram) + "gram")(lef)
        pool_M = TimeDistributed(GlobalMaxPooling1D(), name='TD-GlobalMaxPooling-mid-' + str(n_gram) + "gram")(mid)
        pool_R = TimeDistributed(GlobalMaxPooling1D(), name='TD-GlobalMaxPooling-right-' + str(n_gram) + "gram")(rig)
        convsL.append(pool_L), convsM.append(pool_M), convsR.append(pool_R)

    convoluted_left = Concatenate()(convsL)
    convoluted_mid = Concatenate()(convsM)
    convoluted_right = Concatenate()(convsR)
    # Done making convolution layers

    # To make mid layer like others
    flat_mid = Flatten()(convoluted_mid)

    encode_mid = Dense(300, name='dense-intermediate-mid-encoder')(flat_mid)

    CONV_DIM = sum(conv_hidden_units)

    context_encoder_intermediate1 = Bidirectional(
        LSTM(
            600,
            input_shape=(ONE_SIDE_CONTEXT_SIZE, CONV_DIM),
            recurrent_dropout=0,
            dropout=0,
            return_sequences=True,
            stateful=False
        ),
        name='BiLSTM-context-encoder-intermediate1',
        merge_mode='concat'
    )

    context_encoder = Bidirectional(
        LSTM(
            600,
            input_shape=(ONE_SIDE_CONTEXT_SIZE, CONV_DIM),
            recurrent_dropout=0,
            dropout=0,
            return_sequences=True,
            stateful=False
        ),
        name='BiLSTM-context-encoder',
        merge_mode='concat'
    )

    encode_left = AttentionWithContext()(context_encoder(context_encoder_intermediate1(convoluted_left)))
    encode_right = AttentionWithContext()(context_encoder(context_encoder_intermediate1(convoluted_right)))

    encode_left_drop = Dropout(0.3)(encode_left)
    encode_mid_drop = Dropout(0.2)(encode_mid)
    encode_right_drop = Dropout(0.3)(encode_right)

    encoded_info = Concatenate(axis=-1, name='encode_info')([encode_left_drop, encode_mid_drop, encode_right_drop])

    decoded = Dense(500, name='decoded')(encoded_info)
    decoded_drop = Dropout(0.3, name='decoded_drop')(decoded)

    output = Dense(1, activation='sigmoid')(decoded_drop)
    model = Model(inputs=[left_context, main_input, right_context], outputs=output)

    return model
