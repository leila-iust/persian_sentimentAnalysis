import json
import csv
from hazm import word_tokenize, POSTagger

import utility
import config


def get_dictionary(lexicon_file):
    # lexicon_file = utility.get_file_name(config.data_directory, 'lexicon', lexicon_file)
    with open(lexicon_file, 'r') as f:
        lexicon_dict = json.load(f)

    lexicon_dict_clean = dict()
    for key in lexicon_dict:
        key_clean = utility.clean_comment(key)
        if key_clean != '':
            lexicon_dict_clean[key_clean] = lexicon_dict[key]

    print('lexicon length is {0}'.format(str(len(lexicon_dict))))

    return lexicon_dict_clean


def get_negation():
    # negation_file = utility.get_file_name(config.data_directory, 'lexicon', 'negation.txt')
    negation_file = config.negation_path
    negations = []
    with open(negation_file, 'r')as nf:
        for n in nf:
            negations.append(n.strip())

    return negations


def create_lexicon_input2(opinions, lexicon_file):

    lexicon_dic = get_dictionary(lexicon_file)
    negation = get_negation()

    auxiliary_input = []
    tagger = POSTagger(model='resources/postagger.model')
    for text in opinions:
        sentences = utility.get_sentences(text, tagger)
        score_total = 0
        for idx, sentence in enumerate(sentences):
            if '?' in sentence or '؟' in sentence:
                continue
            if idx == 0 or idx == (len(sentences)-1):
                sw = 1.5
            else:
                sw = 1
            score = 0
            # tokens = word_tokenize(sentence)
            # bigrams = utility.get_bigrams(sentence)

            for l in lexicon_dic:
                if l in sentence:
                    score += lexicon_dic[l]
            for t in negation:
                if t in sentence:
                    score = score * -1
            score_total += score * sw

        auxiliary_input.append(score_total)
    return auxiliary_input


def create_lexicon_input3(opinions, lexicon_file):

    lexicon_dic = get_dictionary(lexicon_file)
    negation = get_negation()

    auxiliary_input = []
    tagger = POSTagger(model='resources/postagger.model')
    for text in opinions:
        sentences = utility.get_sentences(text, tagger)
        score_total = 0
        for sentence in sentences:
            score = 0
            tokens = word_tokenize(sentence)
            bigrams = utility.get_bigrams(sentence)
            bigrams.extend(tokens)
            for token in bigrams:
                if any(token == t for t in negation):
                    score = score * -1
                if token in lexicon_dic:
                    score += lexicon_dic[token]
            score_total += score

        auxiliary_input.append(score_total)
    return auxiliary_input


def create_lexicon_input4(opinions, lexicon_file):

    lexicon_dic = get_dictionary(lexicon_file)
    negation = get_negation()

    auxiliary_input = []
    tagger = POSTagger(model='resources/postagger.model')
    for text in opinions:
        sentences = utility.get_sentences(text, tagger)
        score_total = 0
        for idx, sentence in enumerate(sentences):
            if '?' in sentence or '؟' in sentence:
                continue
            if idx == 0 or idx == (len(sentences)-1):
                sw = 1.5
            else:
                sw = 1
            score = 0
            tokens = word_tokenize(sentence)
            bigrams = utility.get_bigrams(sentence)
            bigrams.extend(tokens)
            for token in bigrams:
                if any(token == t for t in negation):
                    score = score * -1
                if token in lexicon_dic:
                    score += lexicon_dic[token]
            score_total += sw * score

        auxiliary_input.append(score_total)
    return auxiliary_input


def create_lexicon_input(opinions, lexicon_file):

    lexicon_dic = get_dictionary(lexicon_file)
    negation = get_negation()

    auxiliary_input = []
    tagger = POSTagger(model='resources/postagger.model')
    for text in opinions:
        sentences = utility.get_sentences(text, tagger)
        score_total = 0
        for sentence in sentences:
            score = 0
            tokens = word_tokenize(sentence)

            for token in tokens:
                if any(token == t for t in negation):
                    score = score * -1
                if token in lexicon_dic:
                    score += lexicon_dic[token]
            score_total += score

        auxiliary_input.append(score_total)
    return auxiliary_input


def create_lexicon_input_old(opinions, lexicon_file):

    lexicon_dic = get_dictionary(lexicon_file)
    negation = get_negation()

    auxiliary_input = []
    tagger = POSTagger(model='resources/postagger.model')
    for text in opinions:
        score = 0
        tokens = word_tokenize(text)
        for token in tokens:
            # if any(token == t for t in negation):
            #     score = score * -1
            if token in lexicon_dic:
                score += lexicon_dic[token]

        auxiliary_input.append(score)
    return auxiliary_input


def evaluation_lexicon(text, label, lexicon_file):
    ax_input = create_lexicon_input4(text, lexicon_file)
    file_lexicon = utility.get_file_name(config.target_directory, 'lexicon', 'eval_lexicon.csv')
    file_lexicon_mistake = utility.get_file_name(config.target_directory, 'lexicon', 'eval_lexicon_mistake.csv')
    true_count = 0
    total_count = len(text)
    with open(file_lexicon, 'w')as f:
        with open(file_lexicon_mistake, 'w')as fm:
            writer = csv.writer(f)
            writer_fm = csv.writer(fm)
            row = ['Text', 'Prediction', 'True', 'score']
            writer.writerow(row)
            for i, score in enumerate(ax_input):
                polarity = ''
                if score > 0:
                    polarity = 1
                    if label[i] == 1:
                        true_count += 1
                elif score < 0:
                    polarity = 0
                    if label[i] == 0:
                        true_count += 1
                elif score == 0:
                    polarity = 2
                    if label[i] == 2:
                        true_count += 1

                row = [text[i], utility.get_polarity(polarity), utility.get_polarity(label[i]), score]
                writer.writerow(row)
                if utility.get_polarity(polarity) != utility.get_polarity(label[i]):
                    writer_fm.writerow(row)

    print('true_count = {0} and total_count = {1}'.format(str(true_count), str(total_count)))


def create_lexicon(texts, labels):
    pos_text = []
    neg_text = []
    neutral_text = []
    for i, text in enumerate(texts):
        if labels[i] == 1:
            pos_text.append(utility.clean_comment(text))
        elif labels[i] == 0:
            neg_text.append(utility.clean_comment(text))
        elif labels[i] == 2:
            neutral_text.append(utility.clean_comment(text))

    tokenizer_pos = utility.create_tokenizer(pos_text)
    word_index_pos = list(tokenizer_pos.word_index.keys())
    tokenizer_neg = utility.create_tokenizer(neg_text)
    word_index_neg = list(tokenizer_neg.word_index.keys())
    tokenizer_neutral = utility.create_tokenizer(neutral_text)
    word_index_neutral = list(tokenizer_neutral.word_index.keys())

    word_pos = []
    count = 0
    for word in word_index_pos:
        if not any(word == w for w in word_index_neg):
            word_pos.append(word)

    word_neg = []
    for word in word_index_neg:
        if not any(word == w for w in word_index_pos):
            word_neg.append(word)

    word_neutral = []
    for word in word_index_neutral:
        if not any(word == w for w in word_index_pos) and not any(word == w for w in word_index_neg):
            word_neutral.append(word)

    file_lexicon = utility.get_file_name(config.target_directory, 'lexicon', 'perle_cinematicket.json')
    dict_lexicon = dict()

    for word in word_pos:
        dict_lexicon[word] = 1

    for word in word_neg:
        dict_lexicon[word] = -1

    with open(file_lexicon, 'w', encoding='utf8') as fp:
        fp.write(json.dumps(dict_lexicon, indent=4, sort_keys=True, ensure_ascii=False))
        # json.dump(dict_lexicon, fp, ensure_ascii=False)


def merge_lexicons():
    lex1_file = utility.get_file_name(config.data_directory, 'lexicon', 'lexicon_merge.json')
    lex2_file = utility.get_file_name(config.data_directory, 'lexicon', 'lexicon_cinematicket.json')
    # lex3_file = utility.get_file_name(config.data_directory, 'lexicon', 'lexicon_pmi_2411_2.json')
    # lex4_file = utility.get_file_name(config.data_directory, 'lexicon', 'lexicon_pmi_2811.json')

    with open(lex1_file, 'r') as f:
        lex1 = json.load(f)
    with open(lex2_file, 'r') as f:
        lex2 = json.load(f)
    # with open(lex3_file, 'r') as f:
    #     lex3 = json.load(f)
    # with open(lex4_file, 'r') as f:
    #     lex4 = json.load(f)

    lex_final = dict()
    for key in lex1:
        if key not in lex_final:
            lex_final[key] = lex1[key]

    for key in lex2:
        if key not in lex_final:
            lex_final[key] = lex2[key]

    # for key in lex3:
    #     if key not in lex_final:
    #         lex_final[key] = lex3[key]
    #
    # for key in lex4:
    #     if key not in lex_final:
    #         lex_final[key] = lex4[key]

    file_lexicon = utility.get_file_name(config.data_directory, 'lexicon', 'lexicon_merge2.json')
    with open(file_lexicon, 'w', encoding='utf8') as fp:
        fp.write(json.dumps(lex_final, indent=4, sort_keys=True, ensure_ascii=False))

    return lex_final


def lexicon_stat(lexicon_file):
    lexicon_dict = get_dictionary(lexicon_file)

    neg = 0
    pos = 0
    for key in lexicon_dict:
        if lexicon_dict[key] == -1:
            neg += 1
        else:
            pos += 1

    print('lexicon stat:')
    print('negative : {0} and positive : {1}'.format(str(neg), str(pos)))


if __name__ == '__main__':
    # merge_lexicons()
    # t1 = 'با‌رضایت'
    # t2 = 'بارضایت'
    # if t1 == t2:
    #     print('here')

    l_file = utility.get_file_name(config.data_directory, 'lexicon', 'lexicon_merge2.json')
    lexicon_stat(l_file)
    # test = ['خوشگلم']
    # p = create_lexicon_input4(test, l_file)
    # print(str(p))
#     # negation = get_negation()
#     text = 'این فیلم بسیار قشنگ و زیبا بود مخصوصا بازی امیر غفارمنش و هومن حاج عبدالهی اما بازی پژمان بازغی بل بله بلبله شاخ کوتاه اصلا زیبا نبود ترلان پروانه در قسمت قبلی بهتر بازی کرد به جز امیر غفار منش و هومن حاج عبدالهی بقیه بازی ها ضیف بود'
#     opinion_list = []
#     opinion_list.append(text)
#     create_lexicon_input_old(opinion_list, 'perle410.json')
