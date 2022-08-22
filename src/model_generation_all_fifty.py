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


def model_generation_fifty(X_train, y_train):

    #path = fragment_creation.absolute_path + '/512_4/dump'
    save_to_path = fragment_creation.absolute_path + "/512_4/evaluation_data/"

    os.makedirs(save_to_path, exist_ok=True)
    start = time.time()
    #sizes = [20,50,100,150,200]
    sizes = fragment_creation.sizes # sizes = list(np.arange(5, 105, 5)) # vector length
    #sizes = [100]
    no_vocab_cnt = 0 # finds if any model type is missed or not
    f = open('model_gen_time_stat.txt', 'a')
    f.write("vector_size"+"\t"+"model_gen_time"+"\n")

    for i in range(len(sizes)):
        s = time.time()
        size = sizes[i]
        vocab = []
        count = 0

        for file in range(len(X_train)):
            count = count + 1
            data_temp = ''
            for j in range(len(X_train[file])):
                data_temp = data_temp + " " + str(int(X_train[file, j]))
            data_temp = data_temp.strip()
            data_temp = word_tokenize(data_temp)
            vocab.append(data_temp)
            
            if (count % 200) == 0:
                print("Fragment", count, "is processed.\n************************")        
            #model = gensim.models.Word2Vec(vocab, min_count=1)

        model = gensim.models.Word2Vec(vocab, vector_size = size, window=5, min_count=1, workers=4)
        model.wv.save_word2vec_format(save_to_path + "byte2vec_model_vecsize_" + str(size))
        #model.save("test_w2v_forensics_model")
        e = time.time()
        el = e-s
        f.write(str(size)+"\t"+str(el)+"\n")
        print("Done for vector length: ", str(size))
    end = time.time()
    print("Voila! finished building and saving model for fragments!\n")
    if (no_vocab_cnt == 0):
        print("No missed vocabulary\n")
    else:
        print(no_vocab_cnt, " missed vocabualry, please check!\n")    
    print("Time elapsed:", format(round((end-start)/3600,4)), "hours \nTotal files processed:", count)
