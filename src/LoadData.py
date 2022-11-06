import os
import numpy as np
import fragment_creation
import model_generation_all_fifty

#data_512_4 = "/512_4/"
train_base_path = fragment_creation.absolute_path + "/" + fragment_creation.data_set_type + "/"

def load_dataset(data_dir):
    """Loads relevant already prepared FFT-75 dataset"""

    train_data = np.load(os.path.join(data_dir, 'train.npz'), mmap_mode='r')
    train_data = train_data
    x_train, y_train = train_data['x'], train_data['y']
    # one_hot_y_train = to_categorical(y_train)

    print("Training Data loaded with shape: {} and labels with shape - {}".format(x_train.shape,
                                                                                  y_train.shape))
    x_train, y_train = x_train[:100000], y_train[:100000]

    print("Training Data loaded with shape: {} and labels with shape - {}".format(x_train.shape,
                                                                                  y_train.shape))
    val_data = np.load(os.path.join(data_dir, 'val.npz'), mmap_mode='r')
    val_data = val_data
    x_val, y_val = val_data['x'], val_data['y']
    x_val, y_val = x_val[:20000], y_val[:20000]
    # one_hot_y_val = to_categorical(y_val)
    print("Validation Data loaded with shape: {} and labels with shape - {}".format(x_val.shape, y_val.shape))
    test_data = np.load(os.path.join(data_dir, 'test.npz'), mmap_mode='r')
    x_test, y_test = test_data['x'], test_data['y']
    x_test, y_test = x_test[:20000], y_test[:20000]
    # one_hot_y_test = to_categorical(y_val)
    print("Testing Data loaded with shape: {} and labels with shape - {}".format(x_test.shape, y_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

#data2byte - create files with spaces between each char
#word2vec - create files with spaces between each char and convert to word2vec model

if __name__ == "__main__":

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(train_base_path)
    model_generation_all_fifty.model_generation_fifty(X_train, y_train, "_train", model_generation_all_fifty.ModelType.data2byte)
    model_generation_all_fifty.model_generation_fifty(X_val, y_val, "_val", model_generation_all_fifty.ModelType.data2byte)
    model_generation_all_fifty.model_generation_fifty(X_test, y_test, "_test", model_generation_all_fifty.ModelType.data2byte)


