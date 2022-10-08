#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import gensim
import os
import time
import csv
import numpy as np
import fragment_creation
import LoadData

s = '512_4'

saveTrainFeatureAt = fragment_creation.absolute_path + "/" + s + '/train_feature_data/'
saveTestFeatureAt = fragment_creation.absolute_path + "/" + s + '/test_feature_data/'

def generate_features(saveAtPath, X_data, y_data, size, s):
    model_path = fragment_creation.absolute_path + "/" +str(s) + '/evaluation_data'#+'/'

    extlist = ['.0', '.1', '.2', '.3', '.4']#, '.5']
    # extlist = []
    # for i in range(0, 75):
    #     extlist.append('.' + str(i))
    # # print(extlist)

    # load models based on vector length
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path + "/" + "byte2vec_model_vecsize_" + str(size))
    count = 0
    with open(str(saveAtPath) + 'feature_data_vec_' +  '_sample_' + str(size) + '.csv', 'w',
              newline='') as csvfile:
        first_row = []
        for num in range(size):
            first_row.append('feat_' + str(num))
        first_row.append('data_type')
        first_row.append('class_label')

        datawriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        datawriter.writerow(first_row)

        for file in range(len(X_data)):
            count = count + 1
            data_type = str(y_data[file])
            class_label = extlist.index('.' + str(y_data[file]))

            total_vec = size * [0]

           # go byte byte inside the file and provide as input to a model
            curr_file = X_data[file]
            for j in range(len(curr_file)):
                vec = model[str(curr_file[j])]
                total_vec = total_vec + vec
            mean_vec = total_vec / len(curr_file)
            file_data = mean_vec

            file_data = np.append(file_data, data_type)
            file_data = np.append(file_data, class_label)

            datawriter.writerow(file_data)

            if (count % 1000) == 0:
                print("Processing at file:", count)
            # if count == 1000:
            #    break
        print("Feature generated with vector length of " + str(size))

    print('\nData processing done!\n')
    end = time.time()
    print("Time elapsed:", format(round((end - start) / 3600, 4)), "hours\n***************************")

def generateTestFeatures(s, X_test, y_test):
    path = fragment_creation.absolute_path + "/" +str(s) + '/evaluation_data'#+'/'
    start = time.time()
    count = 0

    sizes = fragment_creation.sizes
    for k in sizes:
        size = k
        # extlist = ['.123', '.bin', '.bmp', '.chp', '.csv', '.data', '.dbase3',
        # '.doc', '.docx', '.dwf', '.eps', '.exported', '.f', '.fits', '.fm',
        # '.g3', '.gif', '.gls', '.gz', '.hlp', '.html', '.icns', '.ileaf',
        # '.java', '.jpg', '.js', '.kml', '.kmz', '.lnk', '.log', '.mac', '.odp',
        # '.pdf', '.png', '.pps', '.ppt', '.pptx', '.ps', '.pst', '.pub', '.py',
        # '.rtf', '.sgml', '.sql', '.squeak', '.swf', '.sys', '.tex', '.text',
        # '.tmp', '.troff', '.ttf', '.txt', '.unk', '.vrml', '.wk1', '.wk3',
        # '.wp', '.xbm', '.xls', '.xlsx', '.xml', '.zip']
        # extlist = ['.swf', '.doc', '.ppt', '.pdf', '.html', '.csv', '.xls', '.txt', '.jpg', '.ps',
        #            '.wp', '.rtf', '.unk', '.gif', '.png', '.xml', '.gz', '.log', '.dbase3', '.f', '.', '.java']
        # extlist = ['.swf', '.doc', '.ppt', '.pdf', '.html', '.csv', '.xls', '.txt', '.jpg',
        #            '.rtf',  '.gif', '.png', '.xml', '.gz', '.log']#, '.dbase3', '.f', '.', '.java', '.ps', '.wp', '.unk',]#

        generate_features(saveTestFeatureAt, X_test, y_test, size, s)

def generateTrainFeatures(s, X_train, y_train):
    start = time.time()
    count = 0
    
    sizes = fragment_creation.sizes #list(np.arange(5,105,5))# vector length
    #sizes = [100]
    
    for k in sizes:
        size = k
        generate_features(saveTrainFeatureAt, X_train, y_train, size, s)


if __name__ == "__main__":
    start = time.time()

    os.makedirs(saveTrainFeatureAt, exist_ok=True)
    os.makedirs(saveTestFeatureAt, exist_ok=True)
   # sampleSize = [500, 1000, 1500, 2000, 2500]
    #for s in sampleSize:
    X_train, y_train, X_val, y_val, X_test, y_test = LoadData.load_dataset(LoadData.train_base_path)
    generateTrainFeatures(s, X_train, y_train)
    generateTestFeatures(s, X_test, y_test)
