import pickle
from sklearn.metrics import f1_score
import argparse
from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from numpy import array
import json
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import csv
import pdb

import config
import utility
import lexicon


def max_length(lines):
    return max([len(s.split()) for s in lines])


# def sequence_to_text(list_of_indices):
#     # Looking up words in dictionary
#     words = [reverse_word_map.get(letter) for letter in list_of_indices]
#     str_words = ' '.join([w for w in words if w is not None])
#     return str_words

def sequence_to_text(list_data, reverse_word_map):
    # Looking up words in dictionary
    results = []
    for list_of_indices in list_data:
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        str_words = ' '.join([w for w in words if w is not None])
        results.append(str_words)
    return results


def load_keras_model(model_path):
    loaded_model_one = load_model(model_path)
    return loaded_model_one


def predict_one(loaded_model_one, comment):
    try:
        # loading tokenizer
        tokenizer_path = config.tokenizer_path
        with open(tokenizer_path, 'rb') as handle_one:
            tokenizer_one = pickle.load(handle_one)

        comment = utility.clean_comment(comment)
        print(comment)
        sequences_one = tokenizer_one.texts_to_sequences([comment])
        print(str(sequences_one))
        maxlen_one = config.max_doc_length
        data_one = pad_sequences(sequences_one, maxlen=maxlen_one)

        # new code for f micro
        y_pred_one = loaded_model_one.predict([data_one], verbose=1)
        val_preds_one = np.argmax(y_pred_one, axis=-1)

        sentiment_one = utility.get_polarity(val_preds_one[0])

        return sentiment_one

    except Exception as e:
        return str(e)


def predict_one_lexicon(loaded_model_one, comment):
    try:
        # loading tokenizer
        lexicon_dir = config.lexicon_dir
        tokenizer_path = config.tokenizer_path
        with open(tokenizer_path, 'rb') as handle_one:
            tokenizer_one = pickle.load(handle_one)

        comment = utility.clean_comment(comment)
        print(comment)
        sequences_one = tokenizer_one.texts_to_sequences([comment])
        print(str(sequences_one))
        maxlen_one = config.max_doc_length
        data_one = pad_sequences(sequences_one, maxlen=maxlen_one)

        ax_input_one = lexicon.create_lexicon_input4(comment, lexicon_dir)
        ax_input_one = np.array(ax_input_one)

        # new code for f micro
        y_pred_one = loaded_model_one.predict([data_one, ax_input_one], verbose=1)
        val_preds_one = np.argmax(y_pred_one, axis=-1)

        sentiment_one = utility.get_polarity(val_preds_one[0])

        return sentiment_one

    except Exception as e:
        return str(e)


def predict_and_evaluation(data_file, target_directory):

    model_e = utility.get_file_name(target_directory, 'models', 'model_CNN_1.h5')
    loaded_model_e = load_model(model_e)

    # load filimo data
    results = utility.load_file(data_file)
    texts_e = results['text']
    labels_e = results['label']
    org_texts_e = results['org_text']
    movies_e = results['movie']
    urls_e = 'www.filimo.com'

    # load cinematicket data
    # results_cinematicket = utility.load_file_cinematicket(data_file)
    # texts_e = results_cinematicket['text']
    # labels_e = results_cinematicket['label']
    # org_texts_e = results_cinematicket['org_text']
    # movies_e = results_cinematicket['movie']
    # urls_e = results_cinematicket['url']


    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels_e)
    encoded_label = encoder.transform(labels_e)
    dummy_label = np_utils.to_categorical(encoded_label)
    labels_e = np.array(dummy_label)

    # loading tokenizer
    tokenizer_path_e = utility.get_file_name(target_directory, 'tokenizer', 'tokenizer.pickle')
    with open(tokenizer_path_e, 'rb') as handle_e:
        tokenizer_e = pickle.load(handle_e)

    sequences_e = tokenizer_e.texts_to_sequences(texts_e)
    # maxlen = max_length(texts)
    maxlen_e = config.max_doc_length
    data_e = pad_sequences(sequences_e, maxlen=maxlen_e)
    print('Shape of data tensor:', data_e.shape)

    x_val_e = data_e
    y_val_e = labels_e
    # x_train = data[:-nb_validation_samples]
    # y_train = labels[:-nb_validation_samples]
    # x_val = data[-nb_validation_samples:]
    # y_val = labels[-nb_validation_samples:]

    # score = loaded_model.evaluate(x_val, y_val, verbose=0)
    y_pred_e = loaded_model_e.predict(x_val_e, verbose=1)
    val_preds_e = np.argmax(y_pred_e, axis=-1)
    val_true = np.argmax(y_val_e, axis=-1)

    f_micro = f1_score(array(val_true), array(val_preds_e), average='micro')
    f_macro = f1_score(array(val_true), array(val_preds_e), average='macro')
    print('Test f micro:%f Test f macro: %f' % (round(f_micro * 100, 2), round(f_macro * 100, 2)))

    list_json_e = []
    for j in range(len(org_texts_e)):
        tmp = {}
        value_e = array(val_preds_e)[j]
        sentiment_e = utility.get_polarity(value_e)
        tmp['movie'] = movies_e[j]
        tmp['comment'] = str(org_texts_e[j])
        tmp['polarity'] = sentiment_e
        tmp['source'] = 'deep'
        tmp['url'] = urls_e
        list_json_e.append(tmp)

    results_e = utility.sentiment_output(list_json_e)
    file_result_e = utility.get_file_name(target_directory, 'multi_class', 'results_evaluation.json')
    with open(file_result_e, 'w')as f_e:
        json.dump(results_e, f_e)

    reverse_word_map = dict(map(reversed, tokenizer_e.word_index.items()))
    reverse_data_e = list(sequence_to_text(data_e, reverse_word_map))

    file_result_e = utility.get_file_name(target_directory, 'multi_class', 'results_evaluation_df.csv')
    with open(file_result_e, 'w')as result_f:
        writer = csv.writer(result_f)
        row = ['Tokenized text', 'Text', 'Prediction', 'True']
        writer.writerow(row)

        for i in range(len(org_texts_e)):
            if str(array(val_preds_e)[i]) != str(array(val_true)[i]):
                prediction = utility.get_polarity(int(array(val_preds_e)[i]))
                true = utility.get_polarity(int(array(val_true)[i]))
                row = [reverse_data_e[i], org_texts_e[i], prediction, true]
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='comments sentiment analysis')
    parser.add_argument('--data_file', type=str, help='The path of data file')
    parser.add_argument('--training', type=str, default='', help='The path of train dataset')
    parser.add_argument('--target_directory', type=str, help='The result file path')
    parser.add_argument('--lexicon_dir', type=str, help='The result file path')
    
    args = parser.parse_args()

    model = utility.get_file_name(args.target_directory, 'models', 'model_callback_lexicon_one.h5')
    # model_lexicon_one.h5
    loaded_model = load_model(model)

    # load data from json file
    results_filimo = utility.load_comments2(args.data_file)
    movies = results_filimo['movie']
    texts = results_filimo['text']
    org_texts = results_filimo['org_text']

    # load cinematicket data
    # results_cinema = utility.load_file_cinematicket_all(args.data_file)
    # movies = results_cinema['movie']
    # texts = results_cinema['text']
    # org_texts = results_cinema['org_text']
    
    # auxilary input
    ax_input_test = lexicon.create_lexicon_input4(texts, args.lexicon_dir)
    ax_input_test = np.array(ax_input_test)

    # loading tokenizer
    tokenizer_path = utility.get_file_name(args.target_directory, 'tokenizer', 'tokenizer.pickle')
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    # maxlen = max_length(texts)
    maxlen = config.max_doc_length
    data = pad_sequences(sequences, maxlen=maxlen)
    print('Shape of data tensor:', data.shape)

    x_val = data

    # x_train = data[:-nb_validation_samples]
    # y_train = labels[:-nb_validation_samples]
    # x_val = data[-nb_validation_samples:]
    # y_val = labels[-nb_validation_samples:]

    # score = loaded_model.evaluate(x_val, y_val, verbose=0)
    y_pred = loaded_model.predict([x_val, ax_input_test], verbose=1)
    val_preds = np.argmax(y_pred, axis=-1)

    list_json = []
    if args.training != '':
        texts_manual, labels_manual, org_texts_manual, movies_manual = utility.load_file(args.training)
        for i in range(len(texts_manual)-1):
            temp = {}
            sentiment = utility.get_polarity(labels_manual[i])
            temp['movie'] = movies_manual[i]
            temp['comment'] = org_texts_manual[i]
            temp['polarity'] = sentiment
            temp['source'] = 'manual'
            temp['url'] = 'filimo.com'
            list_json.append(temp)

    for j in range(len(org_texts)):
        temp = {}
        value = array(val_preds)[j]
        sentiment = utility.get_polarity(value)
        temp['movie'] = movies[j]
        temp['comment'] = str(org_texts[j])
        temp['polarity'] = sentiment
        temp['source'] = 'deep'
        temp['url'] = 'filimo.com'
        list_json.append(temp)

    results = utility.sentiment_output(list_json)
    file_result = utility.get_file_name(args.target_directory, 'multi_class', 'results.json')
    with open(file_result, 'w')as f:
        json.dump(results, f)


