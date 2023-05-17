import pickle
from keras.preprocessing.sequence import pad_sequences
import csv
import numpy as np
import argparse
from numpy import array
from pickle import load
from sklearn import preprocessing
from sklearn.metrics import f1_score
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import config
import utility
import embedding
import models


# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN model')
    parser.add_argument('--training', type=str, help='the train dataset file address')
    parser.add_argument('--testing', type=str, default='', help='the test dataset address')
    parser.add_argument('--word2vec', type=str, help='the word2vec file address')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimension of CNN_1 file')
    parser.add_argument('--target_directory', type=str, help='')
    args = parser.parse_args()

    config.target_directory = args.target_directory
    log_directory = 'multi_class'
    file_log = 'log_main.txt'
    file_log = utility.get_file_name(args.target_directory, log_directory, file_log)
    f = open(file_log, 'w')

    # load training dataset
    results = utility.load_file(args.training)
    texts = results['text']
    labels = results['label']
    org_texts = results['org_text']

    results_test = utility.load_file(args.testing)
    text_test = results_test['text']
    label_test = results_test['label']
    org_texts_test = results_test['org_text']

    nb_validation_samples = len(text_test)
    texts.extend(text_test)
    labels.extend(label_test)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_label = encoder.transform(labels)
    dummy_label = np_utils.to_categorical(encoded_label)
    labels = np.array(dummy_label)

    # calculate vocabulary size
    tokenizer = utility.create_tokenizer(texts)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % vocab_size)
    f.writelines('Found %s unique tokens.\n' % vocab_size)
    maxlen = max_length(texts)
    # maxlen = 100
    config.max_doc_length = maxlen
    data = pad_sequences(sequences, maxlen=maxlen)

    # labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    f.writelines('Shape of data tensor:{0}\n'.format(str(data.shape)))
    print('Shape of label tensor:', dummy_label.shape)
    f.writelines('Shape of label tensor:{0}\n'.format(dummy_label.shape))

    # saving
    tokenizer_file = utility.get_file_name(args.target_directory, 'tokenizer', 'tokenizer.pickle')
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # create auxiliary_input
    # ax_input = utility.create_lexicon_input(texts, args.lexicon_dir)
    # if args.normalize:
    #     normalized_ax = preprocessing.normalize([ax_input])
    #     ax_input = normalized_ax.reshape(normalized_ax.shape[1])
    # else:
    #     ax_input = np.array(ax_input)

    # import pdb; pdb.set_trace()
    # split the data into a training set and a validation set

    f.writelines('nb_validation_samples is \n' + str(nb_validation_samples))
    # x_train = data
    # y_train = labels
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]


    # lexicon_train = ax_input[:-nb_validation_samples]
    # lexicon_val = ax_input[-nb_validation_samples:]

    print('Max document length: %d ' % maxlen)
    f.writelines('Max document length: %d \n' % maxlen)
    print('Vocabulary size: %d' % vocab_size)
    f.writelines('Vocabulary size: %d \n' % vocab_size)

    # define model
    # types = ['lexicon', 'lexicon_one', 'concat', 'twitter', 'google', 'random', 'LSTM', 'LSTM_bidirectional', 'concat_glove']
    # types = ['CNN-1', 'random', 'LSTM', 'LSTM_bidirectional']
    embedding_matrix1 = embedding.find_vectors_word2vec(vocab_size, args.word2vec, word_index, args.emb_dim,
                                                        args.target_directory, 'embedding_matrix1')
    # embedding_matrix2 = embedding.find_vectors_word2vec(vocab_size, args.word2vec2, word_index, args.emb_dim2,
    #                                                     args.target_directory, 'embedding_matrix2')
    # load_glove_model = embedding.load_glove_model(args.word2vec3, word_index, args.emb_dim3)

    model = models.define_model_word_embedding_mc(maxlen, vocab_size, embedding_matrix1, args.emb_dim, True, 1)

    # from keras.backend import manual_variable_initialization
    # manual_variable_initialization(True)

    model.fit([x_train], array(y_train), epochs=20, batch_size=256)

    # save the model
    model_directory = 'models'
    file_model = 'model_{0}.h5'.format('CNN_1')
    file_model = utility.get_file_name(args.target_directory, model_directory, file_model)
    model.save(file_model)
    f.writelines(str(model.summary()) + '\n')
    print('model name = {0}'.format('CNN_1'))
    f.writelines('model name = {0} \n'.format('CNN_1'))

    # evaluate model on train dataset
    loss, acc = model.evaluate([x_train], array(y_train), verbose=1)
    result = 'Train Accuracy: %f \n' % (round(acc * 100, 2)) + '\n'
    print('Train Accuracy: %f ' % (round(acc * 100, 2)))

    # evaluate model on test dataset
    loss, acc = model.evaluate([x_val], array(y_val), verbose=1)
    result += 'Test Accuracy:  %f \n' % (round(acc * 100, 2)) + '\n'
    print('Test Accuracy:  %f' % (round(acc * 100, 2)))

    # new code for f micro
    y_pred = model.predict([x_val], verbose=1)
    val_preds = np.argmax(y_pred, axis=-1)
    val_true = np.argmax(y_val, axis=-1)
    f_micro = f1_score(array(val_true), array(val_preds), average='micro')
    f_macro = f1_score(array(val_true), array(val_preds), average='macro')
    result += 'Test f micro:%f Test f macro: %f' % (round(f_micro * 100, 2), round(f_macro * 100, 2)) + '\n'
    print('Test f micro:%f Test f macro: %f' % (round(f_micro * 100, 2), round(f_macro * 100, 2)))

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    reverse_data = list(utility.sequence_to_text(x_val, reverse_word_map))

    result_directory = 'multi_class'
    file_result = 'result_{0}.csv'.format('CNN_1')
    file_result = utility.get_file_name(args.target_directory, result_directory, file_result)

    with open(file_result, 'w')as result_f:
        writer = csv.writer(result_f)
        row = ['Text', 'Prediction', 'True']
        writer.writerow(row)

        for i in range(nb_validation_samples):
            if str(array(val_preds)[i]) != str(array(val_true)[i]):
                prediction = utility.get_polarity(int(array(val_preds)[i]))
                true = utility.get_polarity(int(array(val_true)[i]))
                row = [reverse_data[i], org_texts_test[i], prediction, true]
                writer.writerow(row)

    print(result)
    f.writelines(result)
    f.close()


