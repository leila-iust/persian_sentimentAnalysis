from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, LSTM
from keras.layers import MaxPooling1D
from keras.layers import concatenate
from keras.layers import Bidirectional
from keras import backend as K
import tensorflow as tf
import os

import utility


# Model Hyperparameters
dropout_prob = (0.5, 0.8)
hidden_dims = (50, 100)

filter_sizes = (3, 4, 5)
num_filters = 32


def norm1(embedding1):
    embedding1 = tf.keras.backend.expand_dims(embedding1, axis=-1)

    return embedding1


def norm2(embedding1):
    embedding1 = tf.keras.backend.expand_dims(embedding1, axis=-1)
    embedding2 = tf.keras.backend.expand_dims(embedding1, axis=-1)
    merged = concatenate([embedding1, embedding2], axis=-1)

    return embedding1


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_p = true_positives / (possible_positives + K.epsilon())
        return recall_p

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_p = true_positives / (predicted_positives + K.epsilon())
        return precision_p
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def define_model_concat_mc(length, vocab_size, word_weight_matrix1, word_weight_matrix2, emb_dim1, emb_dim2, trainable1,
                        trainable2, pooling):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, emb_dim1, input_length=length,
                          weights=[word_weight_matrix1], trainable=trainable1)(inputs1)
    # channel 2
    # inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, emb_dim2, input_length=length,
                          weights=[word_weight_matrix2], trainable=trainable2)(inputs1)

    merged = concatenate([embedding1, embedding2], axis=-1)

    z = merged
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1)(z)
        conv = MaxPooling1D(pool_size=pooling)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    drop_out1 = Dropout(dropout_prob[1])(z)
    dense1 = Dense(hidden_dims[0], activation="relu")(drop_out1)
    model_output = Dense(3, activation="softmax")(dense1)

    model_f = Model(inputs=[inputs1], outputs=model_output)
    model_f.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print(model_f.summary())
    pic_directory = 'pic'
    file_pic = 'concat_mc.png'
    file_pic = utility.get_file_name('', pic_directory, file_pic)
    plot_model(model_f, show_shapes=True, to_file=file_pic)
    return model_f


def define_model_M(length, ax_length, vocab_size, word_weight_matrix1, word_weight_matrix2, emb_dim1, emb_dim2, trainable1,
                        trainable2, pooling):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, emb_dim1, input_length=length,
                          weights=[word_weight_matrix1], trainable=trainable1)(inputs1)
    # channel 2
    # inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, emb_dim2, input_length=length,
                          weights=[word_weight_matrix2], trainable=trainable2)(inputs1)

    merged = concatenate([embedding1, embedding2], axis=-1)

    z = merged
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1)(z)
        conv = MaxPooling1D(pool_size=pooling)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    # pooling_ax = AveragePooling1D(pool_size=ax_length)(x)
    auxiliary_input = Input(shape=(ax_length,), name='aux_input')
    x = concatenate([z, auxiliary_input])

    drop_out1 = Dropout(dropout_prob[1])(x)
    dense1 = Dense(hidden_dims[0], activation="relu")(drop_out1)

    model_output = Dense(1, activation="sigmoid")(dense1)

    model_f = Model(inputs=[inputs1, auxiliary_input], outputs=model_output)
    model_f.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", f1])

    print(model_f.summary())
    pic_directory = 'pic'
    file_pic = 'lexicon.png'
    file_pic = utility.get_file_name('', pic_directory, file_pic)
    plot_model(model_f, show_shapes=True, to_file=file_pic)
    return model_f


def define_model_lexicon_mc(length, ax_length, vocab_size, word_weight_matrix1, word_weight_matrix2, emb_dim1, emb_dim2, trainable1,
                        trainable2, pooling):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, emb_dim1, input_length=length,
                          weights=[word_weight_matrix1], trainable=trainable1)(inputs1)
    # channel 2
    # inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, emb_dim2, input_length=length,
                          weights=[word_weight_matrix2], trainable=trainable2)(inputs1)

    merged = concatenate([embedding1, embedding2], axis=-1)

    z = merged
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1)(z)
        conv = MaxPooling1D(pool_size=pooling)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    # pooling_ax = AveragePooling1D(pool_size=ax_length)(x)
    auxiliary_input = Input(shape=(ax_length,), name='aux_input')
    x = concatenate([z, auxiliary_input])

    drop_out1 = Dropout(dropout_prob[1])(x)
    dense1 = Dense(hidden_dims[0], activation="relu")(drop_out1)

    model_output = Dense(3, activation="softmax")(dense1)

    model_f = Model(inputs=[inputs1, auxiliary_input], outputs=model_output)
    model_f.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1])

    print(model_f.summary())
    pic_directory = 'pic'
    file_pic = 'lexicon_mc.png'
    file_pic = utility.get_file_name('', pic_directory, file_pic)
    plot_model(model_f, show_shapes=True, to_file=file_pic)
    return model_f


def define_model_word_embedding_mc(length, vocab_size, word_weight_matrix, emb_dim1, trainable1, pooling):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, emb_dim1, input_length=length,
                          weights=[word_weight_matrix], trainable=trainable1)(inputs1)
    # embedding1 = Embedding(output_dim=emb_dim1, input_length=length,
    #                        weights=[word_weight_matrix], trainable=trainable1)(inputs1)

    z = embedding1
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1)(z)
        conv = MaxPooling1D(pool_size=pooling)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    drop_out1 = Dropout(dropout_prob[1])(z)
    dense1 = Dense(hidden_dims[0], activation="relu")(drop_out1)
    model_output = Dense(3, activation="softmax")(dense1)

    model_f = Model(inputs=[inputs1], outputs=model_output)
    model_f.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print(model_f.summary())
    pic_directory = 'pic'
    file_pic = 'word_embedding_mc.png'
    file_pic = utility.get_file_name(os.getcwd(), pic_directory, file_pic)
    plot_model(model_f, show_shapes=True, to_file=file_pic)
    return model_f


def define_model_lexicon_one_mc(length, vocab_size, word_weight_matrix, emb_dim1, trainable1, pooling):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, emb_dim1, input_length=length,
                          weights=[word_weight_matrix], trainable=trainable1)(inputs1)

    z = embedding1
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1)(z)
        conv = MaxPooling1D(pool_size=pooling)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    auxiliary_input = Input(shape=(1,), name='aux_input')
    x = concatenate([z, auxiliary_input])

    drop_out1 = Dropout(dropout_prob[1])(x)
    dense1 = Dense(hidden_dims[0], activation="relu")(drop_out1)
    model_output = Dense(3, activation="softmax")(dense1)

    model_f = Model(inputs=[inputs1, auxiliary_input], outputs=model_output)
    model_f.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1])

    print(model_f.summary())
    pic_directory = 'pic'
    file_pic = 'lexicon_one_mc.png'
    file_pic = utility.get_file_name('', pic_directory, file_pic)
    plot_model(model_f, show_shapes=True, to_file=file_pic)
    return model_f


def define_model_lstm_mc(length, vocab_size, word_weight_matrix1, emb_dim1, trainable1):
    import pdb
    pdb.set_trace()
    inputs = Input(shape=(length,))
    embeddings = Embedding(vocab_size, emb_dim1, input_length=length, weights=[word_weight_matrix1],
                           trainable=trainable1)(inputs)

    lstm_out = LSTM(length)(embeddings)
    drops = Dropout(dropout_prob[0])(lstm_out)
    outputs = Dense(3, activation='softmax', name='aux_output')(drops)
    models = Model(inputs=[inputs], outputs=outputs)
    # compile
    models.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(models.summary())
    pic_directory = 'pic'
    file_pic = 'LSTM_mc.png'
    file_pic = utility.get_file_name('', pic_directory, file_pic)
    plot_model(models, show_shapes=True, to_file=file_pic)
    return models


def define_model_bidirectional_lstm_mc(length, vocab_size, word_weight_matrix1, emb_dim1, trainable1):
    inputs = Input(shape=(length,))
    embeddings = Embedding(vocab_size, emb_dim1, input_length=length, weights=[word_weight_matrix1],
                           trainable=trainable1)(inputs)

    lstm_out = Bidirectional(LSTM(length))(embeddings)
    drops = Dropout(dropout_prob[0])(lstm_out)
    outputs = Dense(3, activation='softmax', name='aux_output')(drops)
    models = Model(inputs=[inputs], outputs=outputs)
    # compile
    models.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(models.summary())
    pic_directory = 'pic'
    file_pic = 'LSTM_bidirectional_mc.png'
    file_pic = utility.get_file_name('', pic_directory, file_pic)
    plot_model(models, show_shapes=True, to_file=file_pic)
    return models


def define_model_word_embedding_random_mc(length, vocab_size, emb_dim1, trainable1, pooling):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, emb_dim1, input_length=length, trainable=trainable1)(inputs1)

    z = embedding1
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1)(z)
        conv = MaxPooling1D(pool_size=pooling)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    drop_out1 = Dropout(dropout_prob[1])(z)
    dense1 = Dense(hidden_dims[0], activation="relu")(drop_out1)
    model_output = Dense(3, activation="softmax")(dense1)

    model_f = Model(inputs=[inputs1], outputs=model_output)
    model_f.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print(model_f.summary())
    pic_directory = 'pic'
    file_pic = 'random_mc.png'
    file_pic = utility.get_file_name('', pic_directory, file_pic)
    plot_model(model_f, show_shapes=True, to_file=file_pic)
    return model_f






