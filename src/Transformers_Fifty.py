import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import seaborn
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics
import transformers
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.python.data import AUTOTUNE
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFRobertaModel
import json
import matplotlib.pyplot as plt
import random
import seaborn as sn
from src.LoadData import load_dataset, train_base_path
from iteration_utilities import grouper
import csv
import tensorflow.keras.backend as K
from tensorflow import keras
import os
import pandas as pd
import sklearn.utils as sk

tf.debugging.set_log_device_placement(True)
tf.debugging.set_log_device_placement(False)
model_directory = train_base_path + 'history' # directory to save model history after every epoch
model_history_file_name = '/model_history.csv'
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

n_categories = 5
from collections import defaultdict
read_from_path = train_base_path + "/transformers_evaluation_data_small/"
size = 100
tokenizer = AutoTokenizer.from_pretrained("roberta-base") #Tokenizer

iterator_batch_size = 4096
#epoch_batch_size = 10
train_test_ration = 4
def train_roberta_module():
    # detect and init the TPU
    # try:
    #     tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = None)
    #     tf.config.experimental_connect_to_cluster(tpu)
    #     tf.tpu.experimental.initialize_tpu_system(tpu)
    #     tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    # except ValueError:
    #     strategy = tf.distribute.get_strategy()

    batch_size=5#2*tpu_strategy.num_replicas_in_sync#32 * tpu_strategy.num_replicas_in_sync
    print('Batch size:', batch_size)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    X_train,y_train,X_val,y_val,X_test,y_test = load_dataset(train_base_path)

    n_elements=len(X_train)
    print('Elements in dataset:', n_elements)
    categories=sorted(list(set(y_train))) #set will return the unique different entries
    n_categories=len(categories)
    print("{} categories found:".format(n_categories))
    for category in categories:
        print(category)
    return X_train,y_train,X_val,y_val,X_test,y_test , batch_size, n_categories, n_elements



def load_data_from_file(what_data_str, size):
  inputs = ""
  count = 0
  texts = []
  f = open(str(read_from_path) + 'model_' + str(size) + what_data_str , "r")
  while True :
    count += 1
    line = f.readline()
    if not line:# or count > 1000000:
         break
    texts.append(line)
  f.close()
  print(count)
  return texts

def indicize_labels(labels, categories):
    """Transforms string labels into indices"""
    indices=[]
    for j in range(len(labels)):
        for i in range(n_categories):
            if labels[j]==categories[i]:
                indices.append(i)
    return indices



def create_checkpoint_callback(batch_size, my_model):
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "training_cp_50000/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=16* batch_size)
    my_model.save_weights(checkpoint_path.format(epoch=0))
    return cp_callback, checkpoint_dir
# Define a simple sequential model

def create_model():
    #TFRobertaModel.from_pretrained("roberta-base", num_labels=n_categories)#
    model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=n_categories)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1, clipvalue=1),
      #  loss=tf.keras.losses., metrics=['accuracy']
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='Sparse_Top_3_Categorical_Accuracy')],
    )

    return model


def indicize_labels_for_iterator(labels, categories):
    # """Transforms string labels into indices"""
    indices = []
  #  print(len(labels), " ", labels)
    for j in range(len(labels)):
        for i in range(n_categories):
            if labels[j] == categories[i]:
                indices.append(i)

    return indices

def batch_iterator(batch_size, texts):
    for i in grouper(texts, batch_size, truncate=True):
        yield list(i)

def batch_iterator_labels(batch_size, lables, categories):
    for i in grouper(lables, batch_size, truncate=True):
        yield indicize_labels_for_iterator(list(i), categories)

class StoreModelHistory(keras.callbacks.Callback):

  def on_epoch_end(self,batch,logs=None):
    if ('lr' not in logs.keys()):
      logs.setdefault('lr',0)
      logs['lr'] = K.get_value(self.model.optimizer.lr)

    #if(os.path.exists(model_directory)==False):
    #os.system("mkdir " + str(model_directory))
    os.makedirs(model_directory, exist_ok=True)
    if not ('model_history.csv' in os.listdir(model_directory)):
      with open(model_directory + model_history_file_name,'a') as f:
        y=csv.DictWriter(f,logs.keys())
        y.writeheader()

    with open(model_directory + model_history_file_name,'a') as f:
      y=csv.DictWriter(f,logs.keys())
      y.writerow(logs)

def StorePredictionsInFile(pred_hist):

    pred_history_file_name= "/pred_hist.csv"

    with open(model_directory + pred_history_file_name,'a') as f:
        y=csv.writer(f)
        y.writerow(pred_hist)
        return


def predict(model, X_test, y_test, batch_size):
    # trained_model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=n_categories)
    # trained_model.load_weights(train_base_path + '/saved_weights_50.h5')

    print("start predict results")
   # _, _, _, _, X_test, y_test = load_dataset(train_base_path)
    categories = sorted(list(set(y_test)))
    n_categories = len(categories)
    indices = indicize_labels(y_test, categories)  # Integer label indices

    texts = load_data_from_file("_test", size)
    test_batch_size = int(iterator_batch_size / train_test_ration)
    pred_hist = []
    ix = 0
    for test_data, test_lbl in zip(batch_iterator(test_batch_size, texts),
                                    batch_iterator_labels(test_batch_size, y_test, categories)
                                    ):
        tokens = tokenizer(test_data, padding=True, truncation=True, return_tensors='tf')
        dataset_test = tf.data.Dataset.from_tensor_slices((dict(tokens), test_lbl))  # Create a tensorflow dataset

        test_ds = dataset_test.take(test_batch_size).batch(batch_size, drop_remainder=True)

        logits = model.predict(test_ds, verbose=1).logits
        prob = tf.nn.softmax(logits, axis=1).numpy()
        predictions = np.argmax(prob, axis=1)
        pred_hist = np.hstack((pred_hist, predictions))
        StorePredictionsInFile(predictions)
        if (ix%80 == 0):
            print("predict iteration " + str(ix ) + " out of " + str(
                int(len(y_test) / test_batch_size)))
        ix = ix + 1
    truncate_y_test = y_test[:len(pred_hist)]
    confusion_matrix = tf.math.confusion_matrix(truncate_y_test, pred_hist, num_classes=n_categories)

    print("confusion_matrix")
    #plt.figure(figsize=(15, 13))
    seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='seismic', square=True)
    #seaborn.heatmap(confusion_matrix)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()

    fig1.savefig(model_directory + '/predict.png')

    target_names = [np.unique(truncate_y_test)]
    print(target_names, [np.unique(pred_hist)])
    print(classification_report(truncate_y_test, pred_hist))  # , target_names=target_names))

    print("end of execution")

def train(file_name, X_train, y_train, X_val, y_val, batch_size, n_categories, n_elements):

    categories = sorted(list(set(y_train)))
    # indices = indicize_labels_for_iterator(y_train, categories) #Integer label indices
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # Tokenizer
    model = create_model()
    val_test_batch_size = int(iterator_batch_size / train_test_ration)
    texts_data = load_data_from_file("_train", size)
    texts_val = load_data_from_file("_val", size)

    texts_data = sk.shuffle(texts_data, n_samples = len(y_train), random_state=0)
#    print(texts_data, texts_val)
    ix = 0
    history = []
    # Train tokenizer
   # tf.compat.v1.disable_eager_execution()
    for epoch_ix in range(0, 2):
        ix = 0
        for train_data, train_lbl, val_data, val_lbl in zip(batch_iterator(iterator_batch_size, texts_data),
                                                            batch_iterator_labels(iterator_batch_size, y_train,
                                                                                  categories),
                                                            batch_iterator(val_test_batch_size, texts_val),
                                                            batch_iterator_labels(val_test_batch_size, y_val,
                                                                                  categories)
                                                            ):
            print(ix, "  ", train_lbl)
            print(len(train_data), len(train_lbl), len(val_data), len(val_lbl))
            print("1")
           # with tf.device('/device:GPU:0'):
            inputs = tokenizer(train_data, padding=True, truncation=True, return_tensors='tf')
            print("2")
            dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), train_lbl))  # Create a tensorflow dataset
            print("3")
            inputs_val = tokenizer(val_data, padding=True, truncation=True, return_tensors='tf')
            print("4")
            dataset_val = tf.data.Dataset.from_tensor_slices((dict(inputs_val), val_lbl))  # Create a tensorflow dataset

            val_ds = dataset_val.take(val_test_batch_size).batch(batch_size, drop_remainder=True)
            train_ds = dataset.take(iterator_batch_size).batch(batch_size, drop_remainder=True)
            train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

            callback_checkpoint, path_checkpoint = create_checkpoint_callback(batch_size, model)
            print("epoc " + str(epoch_ix) + " iteration " + str(int(ix)) + " out of " + str(int(len(y_train) / iterator_batch_size)) + "  model.fit")
            history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1,
                                callbacks=[callback_checkpoint, StoreModelHistory()])

            ix = ix + 1

    model.save_weights(train_base_path + file_name)
    print("Total train resulting lines ", ix*epoch_ix)
    return ix*epoch_ix


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size, n_categories, n_elements = train_roberta_module()
    file_name = './saved_weights_512_1_full_chunked.h5'
    #EPOCH = 32
  #  X_train, y_train = sk.shuffle(X_train, y_train, random_state=0)

    EPOCH = train(file_name, X_train, y_train, X_val, y_val, batch_size, n_categories, n_elements)


   #model.save('./saved_model.h5') - no
    history_dataframe = pd.read_csv(model_directory + model_history_file_name, sep=',')

    # Plot training & validation loss values
    plt.style.use("ggplot")
    plt.plot(range(1, EPOCH + 1),
             history_dataframe['loss'])
    plt.plot(range(1, EPOCH + 1),
             history_dataframe['val_loss'],
             linestyle='--')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.legend(['Train', 'Val'], loc='upper left')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()

    fig1.savefig(model_directory + '/train.png')
    print("model.predict")

    new_model = create_model()
    new_model.load_weights(train_base_path + file_name)

    predict(new_model, X_test, y_test, batch_size)
    print("model.done")

# def predict(model):
#     # trained_model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=n_categories)
#     # trained_model.load_weights(train_base_path + '/saved_weights_50.h5')
#
#     print("start predict results")
#     _, _, _, _, X_test, y_test = load_dataset(train_base_path)
#     categories = sorted(list(set(y_test)))
#     n_categories = len(categories)
#     indices = indicize_labels(y_test, categories)  # Integer label indices
#
#     texts = load_data_from_file("_test", size)
#
#     tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
#     logits = model.predict(dict(tokens), verbose=1).logits
#     prob = tf.nn.softmax(logits, axis=1).numpy()
#     predictions = np.argmax(prob, axis=1)
#     confusion_matrix = tf.math.confusion_matrix(y_test, predictions, num_classes=n_categories)
#
#     print("confusion_matrix")
#     #plt.figure(figsize=(15, 13))
#     seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='seismic', square=True)
#     #seaborn.heatmap(confusion_matrix)
#     plt.show()
#
#     target_names = [np.unique(y_test)]
#     print(target_names, [np.unique(predictions)])
#     print(classification_report(y_test, predictions))  # , target_names=target_names))
#
#     print("end of execution")


# if __name__ == "__main__":
#     X_train,y_train,X_val,y_val,X_test,y_test , batch_size, n_categories, n_elements = train_roberta_module()
#     categories = sorted(list(set(y_train)))
#     indices = indicize_labels(y_train, categories) #Integer label indices
#     tokenizer = AutoTokenizer.from_pretrained("roberta-base") #Tokenizer
#   #  output_types = {"input_ids": tf.int64, "token_type_ids": tf.int64, "attention_mask": tf.int64}
#
#
#
#     all_train_data = load_data_from_file("_train", size)
#     print("1")
#    # texts = tf.data.Dataset.from_generator(train_dataset_gen, output_types=output_types, args=(all_train_data))
#    # inputs = tokenizer(all_train_data, padding=True, truncation=True, return_tensors='tf') #Tokenize - fails
#     print("2")
#     dataset = tf.data.Dataset.from_tensor_slices((dict(all_train_data), indices)) #Create a tensorflow dataset
#     print("3")
#     categories_val = sorted(list(set(y_val)))
#     print("4")
#     indices_val = indicize_labels(y_val, categories)  # Integer label indices
#     texts_val = load_data_from_file("_val", size)
#
#     inputs_val = tokenizer(texts_val, padding=True, truncation=True, return_tensors='tf')  # Tokenize
#
#     dataset_val = tf.data.Dataset.from_tensor_slices((dict(inputs_val), indices_val))  # Create a tensorflow dataset
#
#     #train test split, we use 10% of the data for validation
#     val_data_size= len(X_val)#int(0.1*n_elements)
#     # val_ds = dataset.take(val_data_size).batch(batch_size, drop_remainder=True)
#     # train_ds = dataset.skip(val_data_size).batch(batch_size, drop_remainder=True)
#     val_ds = dataset_val.take(val_data_size).batch(batch_size, drop_remainder=True)
#     train_ds = dataset.take(n_elements).batch(batch_size, drop_remainder=True)
#     train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
#
#     #with tpu_strategy.scope():
#     # model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=n_categories)
#     # model.compile(
#     #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.),
#     #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     #     metrics=[tf.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='Sparse_Top_3_Categorical_Accuracy')],
#     # )
#     model = create_model()
#
#     callback_checkpoint, path_checkpoint = create_checkpoint_callback(batch_size, model)
#     print("model.fit")
#     history = model.fit(train_ds, validation_data=val_ds, epochs=2, verbose=1, callbacks=[callback_checkpoint])
#     print("model.save")
#
#     model.save_weights(train_base_path +'/saved_weights_512_1_full.h5')
#    #model.save('./saved_model.h5') - no
#
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.ylabel('model loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='best')
#     plt.savefig('train_history.png')
#     plt.show()
#     print("model.predict")
#     new_model = create_model()
#     new_model.load_weights(train_base_path + '/saved_weights_512_1_full.h5')
#
#     predict(new_model)
#     print("model.done")
