#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Simona Lisker
Byte2Vec modeling of the raw frequency distribution for the file fragments
if the vocabulary turns out to be empty, then no model will be generated
-------------------------------------------------------------------------------
    Variables:

        path =  fragments location on which the model will be built
        sizes = vector size starting from 5 to 100 with 5 interval
        output = byte2vec models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import os
from nltk.tokenize import word_tokenize
import gensim
import time
import fragment_creation
import numpy as np
from enum import Enum


class ModelType(Enum):
    word2vec = 0
    data2byte = 1


def create_word_data(X_data, save_to_path, what_data_str):
    vocab = []
    count = 0
    f_model = open(save_to_path + 'model_' + str(100) + what_data_str, 'a')

    for file in range(len(X_data)):
        count = count + 1
        data_temp = ''
        for j in range(len(X_data[file])):
            data_temp = data_temp + " " + str(int(X_data[file, j]))
        data_temp = data_temp.strip()
        # data_temp = word_tokenize(data_temp)
        f_model.write(data_temp + "\n")
        if (count % 200) == 0:
            print("Fragment", count, "is processed.\n************************")
            # model = gensim.models.Word2Vec(vocab, min_count=1)
    f_model.close()
    return count


def format_data(X_data, model_type, size, save_to_path, what_data_str):
    vocab = []
    count = 0

    for file in range(len(X_data)):
        count = count + 1
        data_temp = ''
        for j in range(len(X_data[file])):
            data_temp = data_temp + " " + str(int(X_data[file, j]))
        data_temp = data_temp.strip()
        data_temp = word_tokenize(data_temp)
        vocab.append(data_temp)

        if (count % 200) == 0:
            print("Fragment", count, "is processed.\n************************")
            # model = gensim.models.Word2Vec(vocab, min_count=1)

    model = gensim.models.Word2Vec(vocab, vector_size=size, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format(save_to_path + "byte2vec_model_vecsize_" + str(size))

    # model.save("test_w2v_forensics_model")
    return count


def model_generation_fifty(X_train, y_train, what_data_str, model_type=ModelType.data2byte):
    # path = fragment_creation.absolute_path + '/512_4/dump'
    if (model_type == ModelType.word2vec):
        save_to_path = fragment_creation.absolute_path + '/' + fragment_creation.data_set_type + "/evaluation_data/"
    # elif (ç == ModelType.data2byte):
    else:
        save_to_path = fragment_creation.absolute_path + '/' + fragment_creation.data_set_type + "/transformers_evaluation_data_small/"

    os.makedirs(save_to_path, exist_ok=True)
    start = time.time()
    # sizes = [20,50,100,150,200]
    sizes = fragment_creation.sizes  # sizes = list(np.arange(5, 105, 5)) # vector length
    # sizes = [100]
    no_vocab_cnt = 0  # finds if any model type is missed or not
    f = open('model_gen_time_stat.txt', 'a')
    f.write("vector_size" + "\t" + "model_gen_time" + "\n")
    count = 0
    for i in range(len(sizes)):
        s = time.time()
        size = sizes[i]
        if (model_type == ModelType.word2vec):
            count = format_data(X_train, model_type, size, save_to_path, what_data_str)
        else:
            count = count + create_word_data(X_train, save_to_path, what_data_str)
        e = time.time()
        el = e - s
        f.write(str(size) + "\t" + str(el) + "\n")

        print("Done for vector length: ", str(size), count)
        end = time.time()
        print("Voila! finished building and saving model for fragments!\n")
        if (no_vocab_cnt == 0):
            print("No missed vocabulary\n")
        else:
            print(no_vocab_cnt, " missed vocabualry, please check!\n")
        print("Time elapsed:", format(round((end - start) / 3600, 4)), "hours \nTotal files processed:", count)
