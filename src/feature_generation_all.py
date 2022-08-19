#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:16:11 2017
@author: Md Enamul haque
@purpose: feature generation
@inistitute: University of Louisian at Lafayette
"""
import gensim
import os
import time
import csv
import numpy as np
import fragment_creation

def generateFeatures(s):
    
    path = fragment_creation.absolute_path + "/" +str(s) + '/evaluation_data'#+'/'
    path_fragments = fragment_creation.absolute_path + "/" + str(s) + '/dump'
    start = time.time()
    count = 0
    
    sizes = fragment_creation.sizes #list(np.arange(5,105,5))# vector length
    #sizes = [100]
    
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
        extlist = ['.swf', '.doc', '.ppt', '.pdf', '.html', '.csv', '.xls', '.txt', '.jpg', '.ps',
                   '.wp', '.rtf', '.unk', '.gif', '.png', '.xml', '.gz', '.log', '.dbase3', '.f', '.', '.java']
        
        # load models based on vector length
        model = gensim.models.KeyedVectors.load_word2vec_format(path+ "/"+ "byte2vec_model_vecsize_"+str(size))

        with open(str(saveFeatureAt)+'feature_data_vec_'+str(size)+'_sample_'+str(s)+'.csv', 'w', newline='') as csvfile:
            first_row = []
            for num in range(size):    
                first_row.append('feat_'+ str(num))
            first_row.append('data_type')
            first_row.append('class_label')
            
            
            datawriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            datawriter.writerow(first_row)
            
            for file in os.listdir(path_fragments):
                
                file_data = []
                count = count + 1    
                current = os.path.join(path_fragments, file)
                extension = os.path.splitext(current)[-1]
                fileType = extension[1:].lower()
    
                cur_file = open(current, "rb")
                data = bytearray(cur_file.read())
    
                
                data_type = str(fileType)
                class_label = extlist.index('.'+str(fileType))
                
                total_vec = size * [0]
                
                for i in range(len(data)):
                    vec = model[str(data[i])]
                    total_vec = total_vec + vec
                mean_vec = total_vec / len(data)
                file_data = mean_vec
                    
                file_data = np.append(file_data, data_type)
                file_data = np.append(file_data, class_label)
                
                datawriter.writerow(file_data)
                
                if (count % 100) == 0:
                    print("Processing at file:", count)
                #if count == 1000:
                #    break
            print("Feature generated with vector length of " +str(size))
        
        print('\nData processing done!\n')     
        end = time.time()
        print("Time elapsed:", format(round((end-start)/3600,4)), "hours\n***************************")

if __name__ == "__main__":
    start = time.time()
    s = '512_4'
    global saveFeatureAt
    saveFeatureAt = fragment_creation.absolute_path + "/" +s +'/feature_data/'
    os.makedirs(saveFeatureAt, exist_ok=True)
   # sampleSize = [500, 1000, 1500, 2000, 2500]
    #for s in sampleSize:
    generateFeatures(s)