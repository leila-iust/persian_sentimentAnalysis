import numpy as np
from hazm import word_tokenize
from keras.preprocessing.text import Tokenizer
import os
import re
import string
from emoji import UNICODE_EMOJI
import json
import csv
import pandas
from nltk import bigrams
from numpy import array

import lexicon


def get_polarity(label):
    sentiment = ''
    if label == 0:
        sentiment = 'neg'
    elif label == 1:
        sentiment = 'pos'
    elif label == 2:
        sentiment = 'neutral'

    return sentiment


def clean_emojis(text):
    UNICODE_EMOJI['heart_add'] = '♡'
    for emoji in UNICODE_EMOJI:
        if emoji in text:
            text = text.replace(emoji, ' ' + emoji + ' ')
    return text


def remove_duplicate(word):
    letters = [char for char in word]
    new_word = []
    flag = True
    for i in range(len(letters)):
        first = letters[i]
        if i+1 != len(letters):
            second = letters[i + 1]
        else:
            second = ''
        if flag:
            new_word.append(first)
        if second == first:
            flag = False
        else:
            flag = True
    result = ''.join(l for l in new_word)
    return result


def load_file_to_list_namava_F(text_data_file):
    labels = []
    texts = []
    with open(text_data_file, 'r') as t:
        texts = []  # list of text samples
        labels = []
        for line in t:
            tokens = line.split('\t')
            if len(tokens) <= 2:
                continue

            text = tokens[1]
            # text = clean_comment(text)
            if len(tokens) >= 3 and ('1' in tokens[2] or '0' in tokens[2]):
                sentiment = tokens[2].strip()
            elif len(tokens) >= 4 and ('1' in tokens[3] or '0' in tokens[3]):
                sentiment = tokens[3].strip()
            elif len(tokens) >= 5 and ('0' in tokens[4] or '1' in tokens[4]):
                sentiment = '2'
            else:
                continue

            if '-1' in sentiment:
                label_id = 0
            elif '1' in sentiment:
                label_id = 1
            elif '0' in sentiment:
                label_id = 2

            labels.append(label_id)
            texts.append(text)

    return texts, labels


def load_file_to_list_namava_M(text_data_file):
    with open(text_data_file, 'r') as t:
        texts = []  # list of text samples
        labels_index = {}  # dictionary mapping label name to numeric id
        labels = []
        for line in t:
            tokens = line.split('\t')
            if len(tokens) <= 2:
                continue
            text = tokens[1]
            # text = clean_comment(text)
            if len(tokens) >= 3 and ('1' in tokens[2] or '2' in tokens[2] or '3' in tokens[2]):
                sentiment = tokens[2].strip()
            elif len(tokens) >= 4 and ('1' in tokens[3] or '2' in tokens[3] or '3' in tokens[3]):
                sentiment = '3'
            else:
                continue
            if '1' in sentiment:
                label_id = 1
            elif '2' in sentiment:
                label_id = 0
            elif '3' in sentiment:
                label_id = 2

            labels.append(label_id)
            texts.append(text)

    return texts, labels


def load_file_to_list_namava(file_m, file_f):
    text_f, label_f = load_file_to_list_namava_F(file_f)
    text_m, label_m = load_file_to_list_namava_M(file_m)

    text_f.extend(text_m)
    label_f.extend(label_m)

    texts_duplicate = []
    labels_duplicate = []
    count = 0
    for idx, label in enumerate(label_f):
        tmp = text_f[idx]
        if any(tmp == t for t in texts_duplicate):
            print('delete duplicate: {0}'.format(tmp))
            count += 1
        else:
            if tmp.strip() != '':
                texts_duplicate.append(tmp)
                labels_duplicate.append(label)
    print(str(count))

    count = 0
    texts = []
    labels = []
    for idx, label in enumerate(labels_duplicate):
        if label == 1 and count < 990:
            print('delete: {0}'.format(texts_duplicate[idx]))
            count += 1
        else:
            texts.append(texts_duplicate[idx])
            labels.append(labels_duplicate[idx])
    return texts, labels


def load_file(file_name):
    labels = []
    texts = []
    movies = []
    org_texts = []
    with open(file_name, 'r') as t:
        for line in t:
            tokens = line.split('\t')
            if len(tokens) <= 1:
                continue

            org_text = tokens[0]
            text = clean_comment(org_text)

            if '1' in tokens[1]:
                label_id = 1
            elif '0' in tokens[1]:
                label_id = 0
            elif '2' in tokens[1]:
                label_id = 2

            labels.append(label_id)
            texts.append(text)
            org_texts.append(org_text)
            movies.append(tokens[2])

    results = dict()
    results['text'] = texts
    results['label'] = labels
    results['org_text'] = org_texts
    results['movie'] = movies
    return results


def sequence_to_text(list_data, reverse_word_map):
    # Looking up words in dictionary
    result = []
    for list_of_indices in list_data:
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        str_words = ' '.join([w for w in words if w is not None])
        result.append(str_words)
    return result


def load_file_cinematicket(file_name):
    labels = []
    texts = []
    movies = []
    org_texts = []
    urls = []
    results = dict()
    with open(file_name, 'r') as t:
        for line in t:
            tokens = line.split('\t')
            if len(tokens) <= 1:
                continue

            org_text = tokens[0]
            text = clean_comment(org_text)

            if '1' in tokens[1]:
                label_id = 1
            elif '0' in tokens[1]:
                label_id = 0
            elif '2' in tokens[1]:
                label_id = 2
            else:
                continue

            labels.append(label_id)
            texts.append(text)
            org_texts.append(org_text)
            movies.append(tokens[2])
            urls.append('cinematicket.ir')

    results['text'] = texts
    results['label'] = labels
    results['org_text'] = org_texts
    results['movie'] = movies
    results['url'] = urls
    return results


def create_dataset(texts, labels, movies):
    indices = np.arange(len(texts))
    np.random.shuffle(indices)
    text_shuffle = []
    label_shuffle = []
    movie_shuffle = []
    for i in indices:
        text_shuffle.append(texts[i])
        label_shuffle.append(labels[i])
        movie_shuffle.append(movies[i])

    validation_samples = int(0.2 * len(texts))
    x_train = text_shuffle[:-validation_samples]
    y_train = label_shuffle[:-validation_samples]
    movie_train = movie_shuffle[:-validation_samples]
    x_val = text_shuffle[-validation_samples:]
    y_val = label_shuffle[-validation_samples:]
    movie_val = movie_shuffle[-validation_samples:]

    file_test = get_file_name(os.getcwd(), 'data/dataset', 'namava_test.tsv')
    with open(file_test, 'w')as f_test:
        tsv_writer = csv.writer(f_test, delimiter='\t')
        for i in range(len(x_val) - 1):
            tsv_writer.writerow([x_val[i], y_val[i], movie_val[i]])

    file_train = get_file_name(os.getcwd(),'data/dataset', 'namava_train.tsv')
    with open(file_train, 'w')as f_test:
        tsv_writer = csv.writer(f_test, delimiter='\t')
        for i in range(len(x_train) - 1):
            tsv_writer.writerow([x_train[i], y_train[i], movie_train[i]])

    text_balance, label_balance, movie_balance = balance_dataset(x_train, y_train, movie_train)
    file_train_balance = get_file_name(os.getcwd(), 'data/dataset', 'namava_train_balance.tsv')
    with open(file_train_balance, 'w')as f_test:
        tsv_writer = csv.writer(f_test, delimiter='\t')
        for i in range(len(text_balance) - 1):
            tsv_writer.writerow([text_balance[i], label_balance[i], movie_balance[i]])

    file_all = get_file_name(os.getcwd(), 'data/dataset', 'namava_all.tsv')
    with open(file_all, 'w')as f_all:
        tsv_writer = csv.writer(f_all, delimiter='\t')
        for i in range(len(text_shuffle) - 1):
            tsv_writer.writerow([text_shuffle[i], label_shuffle[i], movie_shuffle[i]])


def balance_dataset(texts, labels, movies):
    neg = 0
    pos = 0
    neutral = 0
    for label in labels:
        if label == 0:
            neg += 1
        elif label == 1:
            pos += 1
        elif label == 2:
            neutral += 1

    min_value = min([neg, pos, neutral])

    text_new = []
    label_new = []
    movie_new = []

    neg = 0
    pos = 0
    neutral = 0
    for i in range(len(texts)):
        value = labels[i]
        flag = False
        if value == 0:
            neg += 1
            if neg <= min_value:
                flag = True
        elif value == 1:
            pos += 1
            if pos <= min_value:
                flag = True
        elif value == 2:
            neutral += 1
            if neutral <= min_value:
                flag = True

        if flag:
            text_new.append(texts[i])
            label_new.append(labels[i])
            movie_new.append(movies[i])

    return text_new, label_new, movie_new


def create_tokenizer(lines):
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(lines)
    return tokenizer


def get_file_name(directory_parent, directory_name, file_name):
    model_directory = os.path.join(directory_parent, directory_name)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    file_name = os.path.join(model_directory, file_name)

    return file_name


def load_comments(file_name, text_index, max_len):
    with open(file_name, 'r')as f:
        texts = []

        for line in f:
            tokens = line.split('\t')
            text = tokens[text_index]
            text = clean_comment(text)

            text_len = len(text.split())
            if text_len <= max_len:
                texts.append(text)

    return texts


def clean_comment(comment):
    try:

        result = remove_duplicate(comment)
        result = clean_emojis(result)
        # remove url
        result = re.sub(r"http\S+", "", result)
        # remove username start with @
        result = re.sub(r"@\S+", "", result)
        # html tags like $amp
        result = re.sub(r'\&\w*;', '', result)
        # result = result.replace("[^a-zA-Z#]", " ")

        result = re.sub(r"[\d]{1,2}:[\d]{1,2} ", " time ", result)
        result = re.sub(r"\d+th ", "no-th ", result)
        result = re.sub("\d+", "", result)

        # remove punctuation escape !?:
        fav_punctuation = string.punctuation.replace('!', '').replace('?', '')
        fav_punctuation += '_%,“«،؛'
        result = re.sub(r'[' + fav_punctuation + ']+', ' ', result)

        # add space before and after a group of !?:
        result = re.sub(r'([!]){2,}', r' <more-exclamation> ', result)
        result = re.sub(r'([?]){2,}', r' <more-question> ', result)
        result = re.sub(r'(!|:|؟|\?)', r' \1 ', result)

        # remove redandant space
        result = re.sub(r'\s{2,}', ' ', result)

        return result.replace('\n', ' ').strip()
    except Exception as e:
        print('exception clean' + str(e))
        return comment


def clean_text(comment):
    try:

        result = remove_duplicate(comment)
        result = clean_emojis(result)
        # remove url
        result = re.sub(r"http\S+", "", result)
        # remove username start with @
        result = re.sub(r"@\S+", "", result)
        # html tags like $amp
        result = re.sub(r'\&\w*;', '', result)
        # result = result.replace("[^a-zA-Z#]", " ")

        result = re.sub(r"[\d]{1,2}:[\d]{1,2} ", " time ", result)
        result = re.sub(r"\d+th ", "no-th ", result)
        result = re.sub("\d+", "", result)

        # remove punctuation escape !?:
        # fav_punctuation = string.punctuation.replace('!', '').replace('?', '')
        # fav_punctuation += '،؛'
        # result = re.sub(r'[' + fav_punctuation + ']+', ' ', result)

        # add space before and after a group of !?:
        result = re.sub(r'([!]){2,}', r' <more-exclamation> ', result)
        result = re.sub(r'([?]){2,}', r' <more-question> ', result)
        result = re.sub(r'(!|:|؟|\?)', r' \1 ', result)

        # remove redandant space
        result = re.sub(r'\s{2,}', ' ', result)

        return result.strip()
    except Exception as e:
        print('exception clean' + str(e))
        return comment


def load_comments2(file_name):

    with open(file_name, 'r') as f:
        distros_dict = json.load(f)
    results_json = {}
    results_movies = []
    results_comments = []
    org_comments = []
    for i in distros_dict:
        title = i['title']
        comments = i['comments']
        for c in comments:
            results_movies.append(title)
            results_comments.append(clean_comment(c['text']))
            org_comments.append(c['text'])

    results_json['movie'] = results_movies
    results_json['text'] = results_comments
    results_json['org_text'] = org_comments
    return results_json


def sentiment_output(resources):
    films = set()

    for d in resources:
        films.add(d['movie'])

    output = []
    for f in films:
        info = dict()
        info['title'] = f
        comments = []
        for r in resources:
            tmp = dict()
            if r['movie'] == f:
                tmp['body'] = r['comment']
                tmp['polarity'] = r['polarity']
                tmp['method'] = r['source']
                tmp['url'] = r['url']
                comments.append(tmp)
        info['comments'] = comments
        output.append(info)

    return output


def map_movie_name(texts, labels, file_movie):
    with open(file_movie, 'r') as f_movie:
        info_dict = json.load(f_movie)

    result_texts = []
    result_movies = []
    result_labels = []

    # import pdb
    # pdb.set_trace()
    for i in range(len(texts)):
        tmp = dict()
        tmp['text'] = texts[i]
        tmp['polarity'] = labels[i]
        tmp['movie'] = 'Unknown'

        for info in info_dict:
            movie_name = info['title']
            comments = info['comments']
            for c in comments:
                if tmp['text'] == c['text']:
                    tmp['movie'] = movie_name

        result_texts.append(tmp['text'])
        result_movies.append(tmp['movie'])
        result_labels.append(tmp['polarity'])

    return result_texts, result_labels, result_movies


def append_column(file_name,file_output, value):
    df = pandas.read_csv(file_name,  sep='\t')
    df['new_column'] = value
    df.to_csv(file_output, sep='\t')


def count_movies(movies):
    tmp = dict()
    for m in movies:
        if m in tmp:
            tmp[m] += 1
        else:
            tmp[m] = 1

    return tmp


def confusion_matrix(file_name):

    true_pos = 0
    true_neg = 0
    true_nu = 0
    false_pos = 0
    false_neg = 0
    false_nu = 0

    with open(file_name, 'r')as f:
        for d in f:
            tokens = d.split(',')
            v_true = tokens[-1]
            v_prediction = tokens[-2]
            if 'pos' in v_true and 'pos' in v_prediction:
                true_pos += 1
            elif 'pos' in v_prediction and 'pos' not in v_true:
                false_pos += 1
            elif 'neg' in v_prediction and 'neg' in v_true:
                true_neg += 1
            elif 'neg' in v_prediction and 'neg' not in v_true :
                false_neg += 1
            elif 'neutral' in v_prediction and 'neutral' in v_true:
                true_nu += 1
            elif 'neutral' in v_prediction and 'neutral' not in v_true:
                false_nu += 1
            else:
                print('v_true: {0}'.format(str(v_true)))
                print('v_prerd: {0}'.format(v_prediction))

    print('true_pos: {0}'.format(str(true_pos)))
    print('true_neg: {0}'.format(str(true_neg)))
    print('true_nu: {0}'.format(str(true_nu)))
    print('false_pos: {0}'.format(str(false_pos)))
    print('false_neg: {0}'.format(str(false_neg)))
    print('false_nu: {0}'.format(str(false_nu)))


def csv_tojson(csv_file, json_file):
    dict_lexicon = dict()
    with open(csv_file, 'r')as csv_f:
        for line in csv_f:
            tokens = line.split('\t')

            if len(tokens)>2:
                try:
                    key = tokens[0].strip()
                    dict_lexicon[key] = float(tokens[2].replace('\n', '').strip())
                except Exception as e:
                    print('{0}:{1}'.format(tokens[0], tokens[2]))
            else:
                print('here')

    with open(json_file, 'w')as json_f:
        json_f.write(json.dumps(dict_lexicon, indent=4, sort_keys=True, ensure_ascii=False))


def create_my_lexicon(csv_file, lexicon_file):
    dict_lexicon = dict()
    with open(csv_file, 'r')as csv_f:
        for line in csv_f:
            tokens = line.split('\t')
            pos = tokens[2]
            neg = tokens[1]
            if pos != '':
                dict_lexicon[pos] = 1
            if neg != '':
                dict_lexicon[neg] = -1

    with open(lexicon_file)as f_json:
        json_data = json.load(f_json)

    for key, value in dict_lexicon.items():
        json_data[key] = value

    with open(lexicon_file, 'w')as json_f:
        json_f.write(json.dumps(json_data, indent=4, sort_keys=True, ensure_ascii=False))


def txt_tojson(txt_folder, json_file):
    dict_lexicon = dict()
    score = 1
    for file_n in os.listdir(txt_folder):
        if 'negative' in file_n:
            score = -1
        else:
            score = 1
        file_name = get_file_name(txt_folder, '', file_n)
        with open(file_name, 'r')as f:
            for line in f:
                key = line.strip()
                dict_lexicon[key] = score

    with open(json_file, 'w')as json_f:
        json_f.write(json.dumps(dict_lexicon, indent=4, sort_keys=True, ensure_ascii=False))


def create_negation():
    file_input = '/home/leila/leila/m_project/sentiment-analysis/data/lexicon/lexicon_manual.tsv'
    file_out_put = '/home/leila/leila/m_project/sentiment-analysis/data/lexicon/negation.txt'

    list_input = []
    with open(file_input, 'r')as input_f:
        for line in input_f:
            tokens = line.split('\t')
            if tokens[0] != '':
                list_input.append(tokens[0])
    with open(file_out_put, 'w')as output_f:
        for l in list_input:
            output_f.writelines(l + '\n')


def get_sentences(text, tagger):
    list_tags = tagger.tag(word_tokenize(text))

    sentences = []
    tmp = ''
    for t in list_tags:
        tag = t[1]
        word = t[0]
        tmp += word + ' '
        if tag == 'V':
            sentences.append(tmp.strip())
            tmp = ''
    try:
        if tmp != '':
            # if list_tags[-1][1] != 'V':
            sentences.append(tmp.strip())
    except:
        print('list tags error')
        if tmp != '':
            sentences.append(tmp.strip())
    return sentences


def get_bigrams(text):
    string_bigrams = bigrams(text.split())

    list_bigrams = []
    for grams in string_bigrams:
        tmp = grams[0] + ' ' + grams[1]
        list_bigrams.append(tmp)

    return list_bigrams


def get_tree(root):
    result = []
    result.append(root)
    for root, dirs, files in os.walk(root):
        for d in dirs:
            result.append(os.path.join(root, d))
        for f in files:
            str_file = os.path.join(root, f)
            if 'venv' not in str_file:
                result.append(os.path.join(root, f))
    return result


if __name__ == "__main__":
    cinematicket_file = '/home/leila/leila/m_project/sentiment-analysis/data/cinematicket/cinematicket_0811.tsv'
    # file_fn = '/home/leila/leila/m_project/sentiment-analysis/data/dataset/Filimo_fn_0201.tsv'
    # file_m = '/home/leila/leila/m_project/sentiment-analysis/data/dataset/filimo-mohsen_0201.tsv'
    # filimo_crawl = '/home/leila/leila/project/namava/data/namava_new/2001-filimo-comments'
    #
    # text1, label1 = load_file_to_list_namava(file_m, file_fn)
    # text_main, label_main, moves_main = map_movie_name(text1, label1, filimo_crawl)
    results_cinematicket = load_file_cinematicket(cinematicket_file)
    # text_main.extend(results_cinematicket['org_text'])
    # label_main.extend(results_cinematicket['label'])
    # moves_main.extend(results_cinematicket['movie'])
    #
    # create_dataset(text_main, label_main, moves_main)
    # file_n = '/home/leila/leila/m_project/sentiment-analysis/data/dataset/namava_train.tsv'
    # text1, label1 = load_file(file_n)
    #
    # results_cinematicket = load_file_cinematicket(cinematicket_file)
    # texts_e = results_cinematicket['text']
    # labels_e = results_cinematicket['label']
    # org_texts_e = results_cinematicket['org_text']
    # movies_e = results_cinematicket['movie']
    # urls_e = results_cinematicket['url']

    # file_n = '/home/leila/leila/m_project/sentiment-analysis/data/dataset/namava_all.tsv'
    # results = load_file(file_n)

    label1 = results_cinematicket['label']

    n = 0
    p = 0
    nu = 0
    for l in label1:
        if l == 0:
            n += 1
        elif l == 1:
            p += 1
        elif l == 2:
            nu += 1

    print('pos = {0} neg = {1} neutral = {2}'.format(str(p), str(n), str(nu)))
    #
    # lexicon_dir = 'perle.json'
    # lexicon.evaluation_lexicon(results['text'], results['label'], lexicon_dir)

    # file_cinema = '
    # /home/leila/leila/m_project/sentiment-analysis/data/cinematicket/cinematicket.tsv'
    # results = load_file_cinematicket(file_cinema)
    # movies = count_movies(results['movie'])
    #
    # n = 0
    # p = 0
    # nu = 0
    label1 = results_cinematicket['label']
    text1 = results_cinematicket['text']
    lexicon.create_lexicon(text1, label1)
    # for l in label1:
    #     if l == 0:
    #         n += 1
    #     elif l == 1:
    #         p += 1
    #     elif l == 2:
    #         nu += 1

    # print('pos = {0} neg = {1} neutral = {2}'.format(str(p), str(n), str(nu)))

    # results_file = '/home/leila/leila/m_project/sentiment-analysis/data/result_CNN-1 (1).csv'
    # confusion_matrix(results_file)

    # csv_fi = '/home/leila/leila/m_project/sentiment-analysis/data/lexicon/PerSent.tsv'
    # txt_fi = '/home/leila/leila/m_project/sentiment-analysis/data/lexicon/Lexicon_based_Sentiment_Analysis/'
    # json_fi = '/home/leila/leila/m_project/sentiment-analysis/data/lexicon/PerLexicon.json'
    # txt_tojson(txt_fi, json_fi)

    # csv_file = '/home/leila/leila/m_project/sentiment-analysis/data/lexicon/lexicon_manual.tsv'
    # create_my_lexicon(csv_file, lexicon_dir)
    # create_negation()
