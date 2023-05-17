import os
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import gensim
import numpy as np
import utility


def find_vectors_word2vec(vocab_size, file_word2vec, word_index, emb_dim, parent_directory, embedding_name):

    log_file = utility.get_file_name(parent_directory, '', 'log_embedding_{0}.txt'.format(str(embedding_name)))
    model = gensim.models.KeyedVectors.load_word2vec_format(file_word2vec, binary=False, unicode_errors='ignore')
    count = 0
    # embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
    embedding_matrix = np.zeros((vocab_size, emb_dim))
    with open(log_file, 'w') as log:
        for word, i in word_index.items():
            try:
                if word in model.vocab.keys():
                    vector = model.wv[word]
                    embedding_matrix[i] = vector
                else:
                    count += 1
                    log.writelines('{0} not in embedding vectors \n'.format(word))
                    embedding_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
            except Exception as e:
                print('Exception happened in word embedding: '+ str(e))
                print('word = ', word)
                pass
        log.writelines('there are {0} words that are not in the embedding {1} \n'.format(str(count), file_word2vec))

    return embedding_matrix


def load_glove_model(file_glove, word_index, emb_dim):
    # import pdb;
    # pdb.set_trace()

    count = 0
    print("Loading Glove Model")
    embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))

    glove_dict = {}
    with open(file_glove, 'r')as glove:
        for line in glove:
            tokens = line.split()
            word_g = tokens[0]
            embedding = np.array([float(val) for val in tokens[1:]])
            glove_dict[word_g] = embedding

    # import pdb;
    # pdb.set_trace()
    for word, i in word_index.items():
        if word in glove_dict.keys():
            embedding_matrix[i] = glove_dict[word]
        else:
            count += 1
            print('{0} not in glove'.format(word))
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

    # import pdb;
    # pdb.set_trace()
    print('there are {0} word that arenot in the embeddind glove'.format(str(count)))
    return embedding_matrix



