#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Simona Lisker
Byte2Vec modeling of the raw frequency distribution for the file fragments
if the vocabulary turns out to be empty, then no model will be generated
-------------------------------------------------------------------------------
    Variables:
    
        path = sample fragments location on which the model will be built
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
#absolute_path = os.path.dirname(os.path.realpath(__file__))
#absolute_path = "/Users/liskers/Documents/Simona/Byte2Vec/"
if __name__ == "__main__":
    
    local = 1
    if local == 1:
        path = fragment_creation.absolute_path + '/512_4/dump'
    else:
        path = './sampled_data/'
    save_to_path = fragment_creation.absolute_path + "/512_4/evaluation_data/"

    os.makedirs(save_to_path, exist_ok=True)
    start = time.time()
    #sizes = [20,50,100,150,200] 
   # sizes = list(np.arange(5, 105, 5)) # vector length
    sizes = [100]
    no_vocab_cnt = 0 # finds if any model type is missed or not
    f = open('model_gen_time_stat.txt', 'a')
    f.write("vector_size"+"\t"+"model_gen_time"+"\n")

    for i in range(len(sizes)):
        s = time.time()
        size = sizes[i]
        vocab = []
        count = 0
        
        for file in os.listdir(path):
            count = count + 1    
            current = os.path.join(path, file)
            extension = os.path.splitext(current)[-1]
            fileType = extension[1:].lower()
            cur_file = open(current, "rb")
            data = bytearray(cur_file.read())
            
            data_temp = ''
            for j in range(len(data)):
                data_temp = data_temp + " " + str(int(data[j]))
            data_temp = data_temp.strip()
            data_temp = word_tokenize(data_temp)
            vocab.append(data_temp)
            
            if (count % 200) == 0:
                print("Fragment", count, "is processed.\n************************")        
            #model = gensim.models.Word2Vec(vocab, min_count=1)

        model = gensim.models.Word2Vec(vocab, vector_size = size, window=5, min_count=1, workers=4) #size=size
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
    #model = gensim.models.Word2Vec(vocab,size=size,window=5,min_count=1,workers=4)
    #model.save_word2vec_format("test_model")
    #new_model = gensim.models.word2vec.Word2Vec.load_word2vec_format("test_model")