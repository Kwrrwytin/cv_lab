import numpy as np
from sklearn.model_selection import train_test_split

def get_data(num, dim, test_num):
    np.random.seed(30)
    xy_set = np.random.uniform(-10, 10, (num, dim))
    f_xy = xy_set[:, 0] ** 2 + xy_set[:, 0] * xy_set[:, 1] + xy_set[:, 1] ** 2
    f_xy = f_xy.reshape(-1, 1)

    xy_train, xy_test, f_train, f_test = train_test_split(xy_set, f_xy, test_size=test_num)
    # print('train shape: {} {}'.format(xy_train.shape, f_train.shape))
    train_data = []
    test_data = []
    for idx in range(xy_train.shape[0]):
        train_data.append((xy_train[idx], f_train[idx]))
    for idx in range(xy_test.shape[0]):
        test_data.append((xy_test[idx], f_test[idx]))
    return train_data, test_data