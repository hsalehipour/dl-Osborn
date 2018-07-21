import numpy as np
import pandas as pd


headers_feature = ['chi', 'eps', 'N2', 'eff']
nz_profile = 512
nfeatures = 2

def load_data(fdir, split=True, ratio=0.85):
    """
    Returns the mixing data-set as (train_x, train_y), (test_x, test_y). where _x and _y denote "features" and "labels"
    respectively.
    """
    data = pd.read_csv(fdir, names=headers_feature)
    eff = data.pop('eff')[0:data.shape[0]:nz_profile]
    chi = data.pop('chi')

    # # normalize the input features
    # data = normalize_data(data, axis=0)
    # xmin = data['eps'].min()
    # xmax = data['eps'].max()
    # data['eps']= (data['eps'] - xmin) / (xmax-xmin)
    # data['N2'] = (data['N2']  - xmin) / (xmax-xmin)

    # data must be reshaped
    features = np.array(data, dtype=np.float32).reshape((-1, nz_profile, nfeatures)).transpose((0,2,1))
    labels = np.array(eff, dtype=np.float32)

    # normalize the input features
    # features = normalize_data(features, axis=2)

    if split:
        # divide dataset (85-15) for training-testing
        train_x, test_x = split_data_train_test(features, ratio)
        train_y, test_y = split_data_train_test(labels  , ratio)
        return (train_x, train_y), (test_x, test_y)
    else:
        return features, labels


def split_data_train_test(data, ratio):
    """
    splits feature or label data into "train" and "test"
    """
    # divide dataset (85-15) for training-testing
    nexample = data.shape[0]
    nexample_train = np.int32(ratio * nexample)

    # split data-set into training and testing set
    train_set = data[:nexample_train]
    test_set = data[nexample_train:]

    return train_set, test_set


def save_data(data_dic,fname):
    """
    "saves the the pands DataFrame into CSV file
    :param fname: file_name
    :param df: DataFrame
    :return:
    """
    df = pd.DataFrame(data=data_dic)
    df.to_csv(fname, sep='\t')
    return