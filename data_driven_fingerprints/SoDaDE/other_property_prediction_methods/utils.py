import numpy as np

def train_test_split(x_train, y_train, i_train = None, percentage=0.8, seed=8):
    # set the random seed
    np.random.seed(seed)

    # obtain number of indexes
    n = x_train.shape[0]

    # randomly shuffle the indexes
    idx = np.arange(n)
    np.random.shuffle(idx)

    # split the indexes into train and test
    split = int(n * percentage)
    train_idx = idx[:split]
    test_idx = idx[split:]

    # re-sort the indexes
    train_idx.sort()
    test_idx.sort()

    # create the train and test sets
    x_train_split = x_train[train_idx]
    y_train_split = y_train[train_idx]
    
    x_test_split = x_train[test_idx]
    y_test_split = y_train[test_idx]

    if i_train is not None:
        i_train_split = i_train[train_idx]
        i_test_split = i_train[test_idx]
        return (x_train_split, i_train_split, y_train_split), (x_test_split, i_test_split, y_test_split)
    else:
        return (x_train_split, y_train_split), (x_test_split, y_test_split)