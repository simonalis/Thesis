import time
import nmslib
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.metrics import confusion_matrix, classification_report
from nltk import word_tokenize
import classification

#M = 15  # ooxml
M = 64  # 64  # pe
efC = 100
print("number of cores:", multiprocessing.cpu_count())
num_threads = multiprocessing.cpu_count()
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
efS = 1000  # pe
query_time_params = {'efSearch': efS}
target_count = 5 #how many classes for final classification
# Number of neighbors
K = 100
space_name = 'l2'#'cosinesimil'####'l2'
def train_zero_shot(features, df_labels, data_type, data_output, output_v_size):


    # Space name should correspond to the space name
    # used for brute-force search
    #space_name = 'l2'  # OOXML

    # Intitialize the library, specify the space, the type of the vector and add data points
    # for l2
    index = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
    index.addDataPointBatch(features)
    # Create an index
    start = time.time()
    index.createIndex(index_time_params, print_progress=True)
    end = time.time()
    print('Index-time parameters', index_time_params)
    print('Indexing time = %f' % (end - start))
    print('Setting query-time parameters', query_time_params)
    index.setQueryTimeParams(query_time_params)

    # Querying
    query_qty = len(features)
    start = time.time()
    # nbrs = index.knnQueryBatch(features, k=K, num_threads=num_threads)
    # end = time.time()
    # print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
    #       (end - start, float(end - start) / query_qty, num_threads * float(end - start) / query_qty))
    print("save model")
    # Save a meta index and data
    name = data_type + "_" + str(output_v_size) + ".bin"
    index_name = data_type + "_" + str(output_v_size) + ".in"
    save_to = data_output + name
    save_to_in = data_output + index_name
    index.saveIndex(save_to, save_data=False)
    # save index labels
    #df_labels[["index", "label"]].to_csv(save_to_in, sep=",")
    pd.DataFrame({"index": range(0, len(df_labels)), "label": df_labels}).to_csv(save_to_in, sep=",")
    print("train save_to bin", save_to)
    print("train save_to in", save_to_in)

def test_zero_shot(X_train, X_test, y_test, data_type, data_output, output_v_size, sample_size, save_at):
    #space_name = 'l2'

    newIndex = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
    # Re-load the index and the data

    # For an optimized L2 index, there's no need to re-load data points, but this would be required for
    # non-optimized index or any other methods different from HNSW (other methods can save only meta indices)
    ############newIndex.addDataPointBatch(X_train)
    name = data_type + "_" + str(output_v_size) + ".bin"
    load_path = data_output + name
    newIndex.loadIndex(load_path, load_data=False)


    index_name = data_type + "_" + str(output_v_size) + ".in"
    print("test read from bin",load_path)
    print("test read from in", data_output + index_name)
    df_labels_train = pd.read_csv(data_output + index_name, sep=",")
    # df_test_labels_fname = data_output + "test_labels" + "_" + str(output_v_size) + ".in"
    # pd.DataFrame({"index": range(0, len(y_test)), "label": y_test}).to_csv(df_test_labels_fname, sep=",")
    # df_labels = pd.read_csv(df_test_labels_fname, sep=",")

    # Setting query-time parameters and querying
    print('Setting query-time parameters', query_time_params)
    newIndex.setQueryTimeParams(query_time_params)
    #K = 10
    ans = []
    for i in range(len(X_test)):
        val = []
        # for ii in range(len(X_test.iloc[i])):
        #     val.append( X_test.iloc[i, ii])
        val = X_test.iloc[i]#X_test.iloc[i]
        label = y_test.iloc[i]#y_test.iloc[i]
        nbrs = newIndex.knnQueryBatch(val.values.reshape(-1, 1), k=K, num_threads=num_threads)#newIndex.knnQueryBatch(val.values.reshape(-1, 1), k=K, num_threads=num_threads)

       # label_index = y_test.loc[[i]]
        for j in range(0, len(nbrs[0][0])):
            find_ix = np.argmin(nbrs[0][1])
            idx = df_labels_train["index"] == nbrs[0][0][find_ix]
            record_train = df_labels_train[idx]

            #record = df_labels[df_labels["index"] == nbrs[0][0][np.argmax(nbrs[0][1])]] # take the highest distance score location and from that gety the index
       #     print("score:", 1 - float(nbrs[0][1][np.argmin(nbrs[0][1])]) / max(nbrs[0][1]), "predicted:",
       #           record.get("label").values[0], "label", label)
            ans.append({"index": i, "label": label, "predicted": record_train.get("label").values[0],
                        "score": 1 - float(nbrs[0][1][find_ix]) / max(nbrs[0][1])})
            break

    df = pd.DataFrame(ans)
    df.to_csv(data_output + index_name + ".csv", sep=",")
    print("test save results", data_output + index_name + ".csv")
    y_t = df["label"]
    y_p = df["predicted"]
    print(np.unique(y_p))
    cnf_matrix = confusion_matrix(y_t, y_p)#, labels=[np.unique(y_t)])
    target_names = [np.unique(y_t)]
    print(target_names, [np.unique(y_p)])
    print(classification_report(y_t, y_p))#, target_names=target_names))

    class_names = np.unique(y_t)
    # cnf_matrix = confusion_matrix(y_test, y_t.values)
    np.set_printoptions(precision=2)
    conf_mat_plot = 0
    if conf_mat_plot == 1:
        classification.plot_conf_matrix_and_save(cnf_matrix, class_names, output_v_size, sample_size, save_at)

    unique, counts = np.unique(y_p, return_counts=True)
    dict_hist = dict(zip(unique, counts))

    for ind in dict_hist.keys():
        if ind == target_count:
            ind= ind-1 # mistake in index value
        #print(target_names[ind], ind)
  #  print(dict_hist)
    return dict_hist

# def pre_process_test_set(data):
#     vocab = []
#     count = 0
#     for file in range(len(data)):
#         count = count + 1
#         data_temp = ''
#         for j in range(len(data[file])):
#             data_temp = data_temp + " " + str(int(data[file, j]))
#         data_temp = data_temp.strip()
#         data_temp = word_tokenize(data_temp)
#         vocab.append(data_temp)
#
#         if (count % 200) == 0:
#             print("Fragment X_test", count, "is processed.\n************************")
#     arr = np.array(vocab)
#     return arr
