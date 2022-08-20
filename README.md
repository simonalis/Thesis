Byte2Vec
============================
This repository contains source codes for file fragment classification using 
<br>Byte2Vec modeling with KNN (modelKNN)
<br>Byte2Vec modeling with ANN (modelNMSLib). 
<br>The code can be used for  
  <br>i).   creating fragment of first 512 from the files, &nbsp
  <br>ii).  generating Byte2Vec models from the fragments, 
 <br>iii). generating features, and 
  <br>iv).  classifying an unknown fragments to its true type.
  <br>v).   This is how the directory structure looks:
<br><img width="430" alt="image" src="https://user-images.githubusercontent.com/104734787/185744015-1f6a0305-1f51-4eb6-8113-077b190e11c2.png">
 <li>512_4/000 should contain original file of different types
  <br><img width="280" alt="image" src="https://user-images.githubusercontent.com/104734787/185744057-cfbf8c5c-6557-4d18-bc73-d7f820a3c2ac.png">
<li>512_4/dump
<br>File fragments of 512, fragment_creation.py
<li>512_4/feature_data
<br>Feature generation data, after execution of feature_generation_all.py
<li>512_4/evaluation_data
<br>model generation results are located here after execution of the model_generation_all.py
<li>512_4/results_data
<br>Test results, after execution of classification.py



