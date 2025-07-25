import pandas as pd
import random
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import preprocessing

# Function to convert a matrix into one-hot encoded format based on percentiles
def to_Onehot_matrix(x, k):
    x = np.array(x)
    max_values = x.max(axis=0)
    min_values = x.min(axis=0)
    non_constant_features = max_values != min_values
    x = x[:, non_constant_features]
    samples_num, features_num = x.shape
    percentiles = np.linspace(0, 100, k + 1)
    segments = np.percentile(x, percentiles, axis=0).T
    data_matrix = np.zeros((samples_num, k, features_num), dtype=int)
    for i in range(samples_num):
        for j in range(features_num):
            if x[i][j] == segments[j][-1]:
                position = k - 1
            else:
                judge1 = x[i][j] >= segments[j]
                judge2 = x[i][j] <= segments[j]
                value1 = np.where(judge1 == True)[0][-1]
                value2 = np.where(judge2 == True)[0][0]
                position = min(value1, value2)
            data_matrix[i, position, j] = 1
    return data_matrix.tolist(), non_constant_features

def generate_dataset_1(samples_num, feature_num, batch_size, k, important_feature_num,deeep_varis):
    # Generate random data for features
    whole_X = np.random.uniform(0, 1, (samples_num, feature_num))
    n = whole_X.shape[0]
    p0 = whole_X.shape[1]
    # Randomly select important features and introduce noise
    art = np.array(random.sample(range(p0), important_feature_num))

    sign = np.random.choice([-1, 1], len(art))
    u = np.random.uniform(0.1, 0.3, len(art))
    u = u * sign
    mat = np.reshape(np.tile(u, int(0.5 * n)), (int(0.5 * n), len(art)))
    whole_X[:int(0.5 * n), art] = whole_X[:int(0.5 * n), art] + mat
    # Generate target labels (binary classification)
    whole_Y = np.zeros((n, 2))
    whole_Y[:int(0.5 * n), 0] = 1
    whole_Y[int(0.5 * n):, 1] = 1
    t = np.random.permutation(n)
    whole_X = whole_X[t]
    whole_Y = whole_Y[t]


    whole_X = preprocessing.scale(whole_X)

    # Split into training, validation, and test sets
    train_Y = whole_Y[:int(0.8 * 0.7 * n)]
    val_Y = whole_Y[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]

    train_X = whole_X[:int(0.8 * 0.7 * n)]
    val_X = whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    # If deeep_varis is true, apply one-hot encoding to the features
    if deeep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)
        whole_X_zeros = np.zeros(p0)
        whole_X_zeros[art] = 1
        whole_X_ones = np.zeros(p0)
        whole_X_ones[non_constant_features] = 1
        index = whole_X_ones * whole_X_zeros
        bool_index = np.zeros((p0), dtype=bool)
        bool_index[non_constant_features] = True
        filtered_index = index[bool_index]
        Mask_art = np.where(filtered_index == 1)[0]

        Mask_train_X = Mask_whole_X[:int(0.8 * 0.7 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset1 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset1 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset1 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=False)
        val_dataloader1 = DataLoader(val_dataset1, batch_size=batch_size, shuffle=False)
        test_dataloader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)

        feature_num = int(sum(non_constant_features))

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader1, val_dataloader1, test_dataloader1, \
            train_X.shape[0], val_X.shape[0], test_X.shape[0], art, Mask_art, p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[0], art, p0, feature_num

def generate_dataset_2(batch_size, k, important_feature_num,deeep_varis):
    # Load MNIST dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_X, train_Y, test_X, test_Y, val_X, val_Y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels, mnist.validation.images, mnist.validation.labels


    train_index = train_Y[:, 0] == 1
    train_X = train_X[train_index]
    test_index = test_Y[:, 0] == 1
    test_X = test_X[test_index]
    val_index = val_Y[:, 0] == 1
    val_X = val_X[val_index]
    p0 = train_X.shape[1]

    n1 = train_X.shape[0]
    n2 = test_X.shape[0]
    n3 = val_X.shape[0]
    t1 = np.random.permutation(n1)
    t2 = np.random.permutation(n2)
    t3 = np.random.permutation(n3)
    train_X = train_X[t1]
    test_X = test_X[t2]
    val_X = val_X[t3]
    # Introduce noise to the selected features
    art = np.array(random.sample(range(p0), important_feature_num))

    sign = np.random.choice([-1, 1], len(art))
    u = np.random.uniform(0.1, 0.3, len(art))
    u = u * sign
    mat1 = np.reshape(np.tile(u, int(0.5 * n1)), (int(0.5 * n1), len(art)))
    mat2 = np.reshape(np.tile(u, int(0.5 * n2)), (int(0.5 * n2), len(art)))
    mat3 = np.reshape(np.tile(u, int(0.5 * n3)), (int(0.5 * n3), len(art)))
    train_X[:int(0.5 * n1), art] = train_X[:int(0.5 * n1), art] + mat1
    test_X[:int(0.5 * n2), art] = test_X[:int(0.5 * n2), art] + mat2
    val_X[:int(0.5 * n3), art] = val_X[:int(0.5 * n3), art] + mat3
    # Create binary labels for classification
    train_Y = np.zeros((n1, 2))
    train_Y[:int(0.5 * n1), 0] = 1
    train_Y[int(0.5 * n1):, 1] = 1
    test_Y = np.zeros((n2, 2))
    test_Y[:int(0.5 * n2), 0] = 1
    test_Y[int(0.5 * n2):, 1] = 1
    val_Y = np.zeros((n3, 2))
    val_Y[:int(0.5 * n3), 0] = 1
    val_Y[int(0.5 * n3):, 1] = 1
    train_X = train_X[t1]
    train_Y = train_Y[t1]

    whole_X = np.row_stack((train_X, test_X, val_X))
    whole_X = preprocessing.scale(whole_X)
    train_X = whole_X[:n1]
    test_X = whole_X[n1:(n1 + n2)]
    val_X = whole_X[-n3:]
    Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)
    feature_num = int(sum(non_constant_features))
    if deeep_varis:
        whole_X_zeros = np.zeros(p0)
        whole_X_zeros[art] = 1
        whole_X_ones = np.zeros(p0)
        whole_X_ones[non_constant_features] = 1
        index = whole_X_ones * whole_X_zeros
        bool_index = np.zeros((p0), dtype=bool)
        bool_index[non_constant_features] = True
        filtered_index = index[bool_index]
        Mask_art = np.where(filtered_index == 1)[0]

        Mask_train_X = Mask_whole_X[:n1]
        Mask_test_X = Mask_whole_X[n1:(n1 + n2)]
        Mask_val_X = Mask_whole_X[-n3:]
        train_dataset2 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset2 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset2 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))
        train_dataloader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
        val_dataloader2 = DataLoader(val_dataset2, batch_size=batch_size, shuffle=True)
        test_dataloader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=True)


        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader2, val_dataloader2, test_dataloader2, n1, n3, n2, art, Mask_art, p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[0], art, p0, feature_num


def generate_dataset_3(samples_num, feature_num, batch_size, k, important_feature_num,deeep_varis):
    lb, ub = 0.8, 1

    whole_X = np.random.uniform(0, 1, (samples_num, feature_num))
    n = whole_X.shape[0]
    p0 = whole_X.shape[1]

    art = np.random.choice(p0, important_feature_num, replace=False)

    sign = np.random.choice([-1, 1], (int(0.5 * n), len(art)))
    u = np.random.uniform(lb, ub, (int(0.5 * n), len(art)))
    u = u * sign
    whole_X[:int(0.5 * n), art] = whole_X[:int(0.5 * n), art] + u

    whole_Y = np.zeros((n, 2))
    whole_Y[:int(0.5 * n), 0] = 1
    whole_Y[int(0.5 * n):, 1] = 1
    t = np.random.permutation(n)
    whole_X = whole_X[t]
    whole_Y = whole_Y[t]

    train_Y = whole_Y[:int(0.8 * 0.7 * n)]
    val_Y = whole_Y[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]

    whole_X = preprocessing.scale(whole_X)
    train_X = whole_X[:int(0.8 * 0.7 * n)]
    val_X = whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    pd.DataFrame(whole_X).to_csv("X.csv", index=False)
    pd.DataFrame(whole_Y).to_csv("Y.csv", index=False)
    if deeep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)
        reshaped_mask = np.array(Mask_whole_X).reshape(len(Mask_whole_X), -1)
        pd.DataFrame(reshaped_mask).to_csv("Mask_X.csv", index=False)
        whole_X_zeros = np.zeros(p0)
        whole_X_zeros[art] = 1
        whole_X_ones = np.zeros(p0)
        whole_X_ones[non_constant_features] = 1
        index = whole_X_ones * whole_X_zeros
        bool_index = np.zeros((p0), dtype=bool)
        bool_index[non_constant_features] = True
        filtered_index = index[bool_index]
        Mask_art = np.where(filtered_index == 1)[0]

        Mask_train_X = Mask_whole_X[:int(0.8 * 0.7 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]
        train_dataset3 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset3 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset3 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader3 = DataLoader(train_dataset3, batch_size=batch_size, shuffle=True)
        val_dataloader3 = DataLoader(val_dataset3, batch_size=batch_size, shuffle=True)
        test_dataloader3 = DataLoader(test_dataset3, batch_size=batch_size, shuffle=True)

        feature_num = int(sum(non_constant_features))

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader3, val_dataloader3, test_dataloader3, \
        train_X.shape[0], val_X.shape[0], test_X.shape[0], art, Mask_art, p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[0], art, p0, feature_num
def generate_dataset_4(samples_num, feature_num, batch_size, k, important_feature_num,deeep_varis):
    whole_X = np.random.uniform(-1, 1, (samples_num, feature_num))
    n = whole_X.shape[0]
    p0 = whole_X.shape[1]

    art = np.array(random.sample(range(p0), important_feature_num))

    sign = np.random.choice([-1, 1], len(art))
    beta = np.random.uniform(1, 3, len(art))
    beta = beta * sign
    epsilon = np.random.randn(n)

    wholenew_X = np.zeros((n, len(art)))
    wholenew_X[:, :16] = whole_X[:, art[:16]]
    wholenew_X[:, 16:32] = np.sin(whole_X[:, art[16:32]])
    wholenew_X[:, 32:48] = np.exp(whole_X[:, art[32:48]])
    wholenew_X[:, 48:] = np.maximum(whole_X[:, art[48:]], np.zeros((n, 16)))

    sign2 = np.random.choice([-1, 1], 4)
    beta2 = np.random.uniform(1, 3, 4)
    beta2 = beta2 * sign2

    wholenew_X2 = np.zeros((n, 4))
    wholenew_X2[:, 0] = whole_X[:, art[14]] * whole_X[:, art[15]]
    wholenew_X2[:, 1] = whole_X[:, art[30]] * whole_X[:, art[31]]
    wholenew_X2[:, 2] = whole_X[:, art[46]] * whole_X[:, art[47]]
    wholenew_X2[:, 3] = whole_X[:, art[62]] * whole_X[:, art[63]]


    whole_Y = np.dot(wholenew_X, beta) + np.dot(wholenew_X2, beta2) + epsilon
    whole_Y = np.reshape(whole_Y, (n, 1))

    train_Y = whole_Y[:int(0.8 * 0.7 * n)]
    val_Y = whole_Y[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]

    whole_X = preprocessing.scale(whole_X)
    train_X = whole_X[:int(0.8 * 0.7 * n)]
    val_X = whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    if deeep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)
        whole_X_zeros = np.zeros(p0)
        whole_X_zeros[art] = 1
        whole_X_ones = np.zeros(p0)
        whole_X_ones[non_constant_features] = 1
        index = whole_X_ones * whole_X_zeros
        bool_index = np.zeros((p0), dtype=bool)
        bool_index[non_constant_features] = True
        filtered_index = index[bool_index]
        Mask_art = np.where(filtered_index == 1)[0]
    
        Mask_train_X = Mask_whole_X[:int(0.8 * 0.7 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]
    
        train_dataset4 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset4 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset4 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))
    
        train_dataloader4 = DataLoader(train_dataset4, batch_size=batch_size, shuffle=True)
        val_dataloader4 = DataLoader(val_dataset4, batch_size=batch_size, shuffle=True)
        test_dataloader4 = DataLoader(test_dataset4, batch_size=batch_size, shuffle=True)
    
        feature_num = int(sum(non_constant_features))
    
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader4, val_dataloader4, test_dataloader4, \
        train_X.shape[0], val_X.shape[0], test_X.shape[0], art, Mask_art, p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y,train_X.shape[0], val_X.shape[0], test_X.shape[0],art,p0, feature_num

def generate_dataset_5(samples_num, feature_num, batch_size, k, important_feature_num,deeep_varis):
    whole_X = np.random.uniform(-1, 1, (samples_num, feature_num))
    n = whole_X.shape[0]
    p0 = whole_X.shape[1]

    art = np.array(random.sample(range(p0), important_feature_num))  

    sign = np.random.choice([-1, 1], len(art))
    beta = np.random.uniform(1, 3, len(art))
    beta = beta * sign
    epsilon = np.random.randn(n)*2

    wholenew_X = np.zeros((n, len(art)))
    wholenew_X[:, :16] = whole_X[:, art[:16]]
    wholenew_X[:, 16:32] = np.sin(whole_X[:, art[16:32]])
    wholenew_X[:, 32:48] = np.exp(whole_X[:, art[32:48]])
    wholenew_X[:, 48:] = np.maximum(whole_X[:, art[48:]], np.zeros((n, 16)))

    sign2 = np.random.choice([-1, 1], 4)
    beta2 = np.random.uniform(1, 3, 4)
    beta2 = beta2 * sign2

    wholenew_X2 = np.zeros((n, 4))
    wholenew_X2[:, 0] = whole_X[:, art[14]] * whole_X[:, art[15]]
    wholenew_X2[:, 1] = whole_X[:, art[30]] * whole_X[:, art[31]]
    wholenew_X2[:, 2] = whole_X[:, art[46]] * whole_X[:, art[47]]
    wholenew_X2[:, 3] = whole_X[:, art[62]] * whole_X[:, art[63]]


    whole_Y = np.dot(wholenew_X, beta) + np.dot(wholenew_X2, beta2) + epsilon
    whole_Y = np.reshape(whole_Y, (n, 1))

    train_Y = whole_Y[:int(0.8 * 0.7 * n)]
    val_Y = whole_Y[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]

    whole_X = preprocessing.scale(whole_X)
    train_X = whole_X[:int(0.8 * 0.7 * n)]
    val_X = whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    if deeep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)
        whole_X_zeros = np.zeros(p0)
        whole_X_zeros[art] = 1
        whole_X_ones = np.zeros(p0)
        whole_X_ones[non_constant_features] = 1
        index = whole_X_ones * whole_X_zeros
        bool_index = np.zeros((p0), dtype=bool)
        bool_index[non_constant_features] = True
        filtered_index = index[bool_index]
        Mask_art = np.where(filtered_index == 1)[0]

        Mask_train_X = Mask_whole_X[:int(0.8 * 0.7 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset4 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset4 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset4 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader4 = DataLoader(train_dataset4, batch_size=batch_size, shuffle=True)
        val_dataloader4 = DataLoader(val_dataset4, batch_size=batch_size, shuffle=True)
        test_dataloader4 = DataLoader(test_dataset4, batch_size=batch_size, shuffle=True)

        feature_num = int(sum(non_constant_features))

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader4, val_dataloader4, test_dataloader4, \
        train_X.shape[0], val_X.shape[0], test_X.shape[0], art, Mask_art, p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[0],art, p0, feature_num

def generate_dataset_6(samples_num, feature_num, batch_size, k, important_feature_num,deeep_varis):
    whole_X = np.random.uniform(-1, 1, (samples_num, feature_num))
    n = whole_X.shape[0]
    p0 = whole_X.shape[1]

    art = np.array(random.sample(range(p0), important_feature_num))

    sign = np.random.choice([-1, 1], len(art))
    beta = np.random.uniform(1, 3, len(art))
    beta = beta * sign
    epsilon = np.random.randn(n)

    wholenew_X = np.zeros((n, len(art)))
    wholenew_X[:, :16] = whole_X[:, art[:16]]
    wholenew_X[:, 16:32] = 2*np.sin(whole_X[:, art[16:32]])
    wholenew_X[:, 32:48] = 2*np.exp(whole_X[:, art[32:48]])
    wholenew_X[:, 48:] = 2*np.maximum(whole_X[:, art[48:]], np.zeros((n, 16)))

    sign2 = np.random.choice([-1, 1], 4)
    beta2 = np.random.uniform(1, 3, 4)
    beta2 = beta2 * sign2

    wholenew_X2 = np.zeros((n, 4))
    wholenew_X2[:, 0] = whole_X[:, art[14]] * whole_X[:, art[15]]
    wholenew_X2[:, 1] = whole_X[:, art[30]] * whole_X[:, art[31]]
    wholenew_X2[:, 2] = whole_X[:, art[46]] * whole_X[:, art[47]]
    wholenew_X2[:, 3] = whole_X[:, art[62]] * whole_X[:, art[63]]


    whole_Y = np.dot(wholenew_X, beta) + np.dot(wholenew_X2, beta2) + epsilon
    whole_Y = np.reshape(whole_Y, (n, 1))

    train_Y = whole_Y[:int(0.8 * 0.7 * n)]
    val_Y = whole_Y[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]

    whole_X = preprocessing.scale(whole_X)
    train_X = whole_X[:int(0.8 * 0.7 * n)]
    val_X = whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    if deeep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)
        whole_X_zeros = np.zeros(p0)
        whole_X_zeros[art] = 1
        whole_X_ones = np.zeros(p0)
        whole_X_ones[non_constant_features] = 1
        index = whole_X_ones * whole_X_zeros
        bool_index = np.zeros((p0), dtype=bool)
        bool_index[non_constant_features] = True
        filtered_index = index[bool_index]
        Mask_art = np.where(filtered_index == 1)[0]

        Mask_train_X = Mask_whole_X[:int(0.8 * 0.7 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset4 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset4 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset4 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader4 = DataLoader(train_dataset4, batch_size=batch_size, shuffle=True)
        val_dataloader4 = DataLoader(val_dataset4, batch_size=batch_size, shuffle=True)
        test_dataloader4 = DataLoader(test_dataset4, batch_size=batch_size, shuffle=True)

        feature_num = int(sum(non_constant_features))

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader4, val_dataloader4, test_dataloader4, \
        train_X.shape[0], val_X.shape[0], test_X.shape[0], art, Mask_art, p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y,train_X.shape[0], val_X.shape[0], test_X.shape[0], art, p0, feature_num

def load_rna(batch_size, k,deeep_varis):
    whole_X = np.loadtxt("Datasets/data/rna/chen_X.txt")
    whole_Y = np.loadtxt("Datasets/data/rna/chen_Y.txt")
    whole_X = whole_X.T

    n = whole_X.shape[0]
    t = np.random.permutation(n)
    whole_X = whole_X[t]
    whole_Y = whole_Y[t]
    p0 = whole_X.shape[1]

    train_Y = whole_Y[:int(0.8 * 0.7 * n)]
    val_Y = whole_Y[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]
    whole_X = preprocessing.scale(whole_X)

    train_X = whole_X[:int(0.8 * 0.7 * n)]
    val_X = whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    if deeep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)
        Mask_train_X = Mask_whole_X[:int(0.8 * 0.7 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset6 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset6 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset6 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader6 = DataLoader(train_dataset6, batch_size=batch_size, shuffle=True)
        val_dataloader6 = DataLoader(val_dataset6, batch_size=batch_size, shuffle=True)
        test_dataloader6 = DataLoader(test_dataset6, batch_size=batch_size, shuffle=True)
        feature_num = int(sum(non_constant_features))

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader6, val_dataloader6, test_dataloader6, \
        train_X.shape[0], val_X.shape[0], test_X.shape[0], p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y,train_X.shape[0], val_X.shape[0], test_X.shape[0], p0


def load_dream(batch_size, k,deeep_varis):

    data_path = 'Datasets/data/Dream/'

    y_train = pd.read_csv(data_path + "Preterm.csv", index_col=0).was_preterm
    y_train = y_train.astype(int)
    taxo_train = pd.read_csv(data_path + "Taxonomy.csv", index_col=0)
    phylo_train = pd.read_csv(data_path + "Phylotype.csv", index_col=0)

    train_data_dict = {
        "Phylotype": phylo_train,
        "Taxonomy": taxo_train
    }
    y_train = y_train.astype(int)

    for name, df in train_data_dict.items():
        df.columns = df.columns.str.replace("/", "_", regex=True)
        train_data_dict[name] = df
    X_tot = pd.concat(train_data_dict.values(), axis="columns")
    feature_names = X_tot.columns

    scaler = StandardScaler()
    X_tot = scaler.fit_transform(X_tot)
    feature_num = X_tot.shape[1]
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_tot, y_train, test_size=0.3, stratify=y_train)

    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp)
    X_train, X_val, X_test = X_train.tolist(), X_val.tolist(), X_test.tolist()
    train_samples_num, val_samples_num, test_samples_num = len(X_train), len(X_val), len(X_test)
    Y_train, Y_val, Y_test = Y_train.values.tolist(), Y_val.values.tolist(), Y_test.values.tolist()

    train_X = np.array(X_train)
    val_X = np.array(X_val)
    test_X = np.array(X_test)
    Y_train = torch.tensor(Y_train, dtype=torch.int64)
    Y_train = F.one_hot(Y_train, num_classes=2).squeeze(1).float().double()
    Y_val = torch.tensor(Y_val, dtype=torch.int64)
    Y_val = F.one_hot(Y_val, num_classes=2).squeeze(1).float().double()
    Y_test = torch.tensor(Y_test, dtype=torch.int64)
    Y_test = F.one_hot(Y_test, num_classes=2).squeeze(1).float().double()

    train_Y = Y_train.numpy()
    val_Y = Y_val.numpy()
    test_Y = Y_test.numpy()
    p0 = train_X.shape[1]
    if deeep_varis:
        X_train, non_constant_features = to_Onehot_matrix(X_train, k)
        X_val, _ = to_Onehot_matrix(X_val, k)
        X_test, _ = to_Onehot_matrix(X_test, k)

        train_dataset = TensorDataset(torch.tensor(X_train).unsqueeze(1), Y_train)
        val_dataset = TensorDataset(torch.tensor(X_val).unsqueeze(1), Y_val)
        test_dataset = TensorDataset(torch.tensor(X_test).unsqueeze(1), Y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader, val_dataloader, test_dataloader, train_samples_num, val_samples_num, test_samples_num, feature_num, p0, feature_names
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y,train_samples_num, val_samples_num, test_samples_num, p0, feature_names

def load_er(batch_size,k,deep_varis):
    X=pd.read_csv("Datasets/data/BC/matrix.csv")
    Y=pd.read_csv("Datasets/data/BC/meta.csv")

    X=X.T
    Y.index = X.index
    tnbc_idx = Y['cancer'] == 'ER+'
    idx_list = tnbc_idx[tnbc_idx].index

    X_tnbc = X.loc[idx_list]
    Y_tnbc = Y.loc[idx_list]
    n = len(Y_tnbc)
    whole_Y = np.zeros((n, 2))
    for i in range(n):
        if Y_tnbc.iloc[i,0] == 'Pre':
            whole_Y[i,0] = 1
        else:
            whole_Y[i, 1] = 1
    whole_X = preprocessing.scale(X_tnbc.values)
    t = np.random.permutation(n)
    whole_X = whole_X[t]
    whole_Y = whole_Y[t]
    train_Y = whole_Y[:int(0.8 * 0.9 * n)]
    val_Y = whole_Y[int(0.8 * 0.9 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]

    train_X = whole_X[:int(0.8 * 0.9 * n)]
    val_X = whole_X[int(0.8 * 0.9 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    p0=whole_X.shape[1]
    if deep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)


        Mask_train_X = Mask_whole_X[:int(0.8 * 0.9 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.9 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        feature_num = int(sum(non_constant_features))

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader, val_dataloader, test_dataloader, \
            train_X.shape[0], val_X.shape[0], test_X.shape[0], p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[0], p0


def load_her2(batch_size,k,deep_varis):
    X=pd.read_csv("Datasets/data/BC/matrix.csv")
    Y=pd.read_csv("Datasets/data/BC/meta.csv")

    X=X.T
    Y.index = X.index
    tnbc_idx = Y['cancer'] == 'HER2+'
    idx_list = tnbc_idx[tnbc_idx].index

    X_tnbc = X.loc[idx_list]
    Y_tnbc = Y.loc[idx_list]
    n = len(Y_tnbc)
    whole_Y = np.zeros((n, 2))
    for i in range(n):
        if Y_tnbc.iloc[i,0] == 'Pre':
            whole_Y[i,0] = 1
        else:
            whole_Y[i, 1] = 1
    whole_X = preprocessing.scale(X_tnbc.values)
    t = np.random.permutation(n)
    whole_X = whole_X[t]
    whole_Y = whole_Y[t]
    train_Y = whole_Y[:int(0.8 * 0.9 * n)]
    val_Y = whole_Y[int(0.8 * 0.9 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]

    train_X = whole_X[:int(0.8 * 0.9 * n)]
    val_X = whole_X[int(0.8 * 0.9 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    p0=whole_X.shape[1]
    if deep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)


        Mask_train_X = Mask_whole_X[:int(0.8 * 0.9 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.9 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        feature_num = int(sum(non_constant_features))

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader, val_dataloader, test_dataloader, \
            train_X.shape[0], val_X.shape[0], test_X.shape[0], p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[0], p0


def load_tnbc(batch_size,k,deep_varis):
    X=pd.read_csv("Datasets/data/BC/matrix.csv")
    Y=pd.read_csv("Datasets/data/BC/meta.csv")

    X=X.T
    Y.index = X.index
    tnbc_idx = Y['cancer'] == 'TNBC'
    idx_list = tnbc_idx[tnbc_idx].index

    X_tnbc = X.loc[idx_list]
    Y_tnbc = Y.loc[idx_list]
    n = len(Y_tnbc)
    whole_Y = np.zeros((n, 2))
    for i in range(n):
        if Y_tnbc.iloc[i,0] == 'Pre':
            whole_Y[i,0] = 1
        else:
            whole_Y[i, 1] = 1
    whole_X = preprocessing.scale(X_tnbc.values)
    t = np.random.permutation(n)
    whole_X = whole_X[t]
    whole_Y = whole_Y[t]
    train_Y = whole_Y[:int(0.8 * 0.9 * n)]
    val_Y = whole_Y[int(0.8 * 0.9 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]

    train_X = whole_X[:int(0.8 * 0.9 * n)]
    val_X = whole_X[int(0.8 * 0.9 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    p0=whole_X.shape[1]
    if deep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)


        Mask_train_X = Mask_whole_X[:int(0.8 * 0.9 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.9 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        feature_num = int(sum(non_constant_features))

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader, val_dataloader, test_dataloader, \
            train_X.shape[0], val_X.shape[0], test_X.shape[0], p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[0], p0


def generate_dataset_1_DL(samples_num, feature_num, batch_size, k, important_feature_num, feature_index,
                            deep_varis):
    whole_X = np.random.uniform(0, 1, (samples_num, feature_num))
    n = whole_X.shape[0]
    p0 = whole_X.shape[1]

    art = np.array(random.sample(range(p0), important_feature_num))

    sign = np.random.choice([-1, 1], len(art))
    u = np.random.uniform(0.1, 0.3, len(art))
    u = u * sign
    mat = np.reshape(np.tile(u, int(0.5 * n)), (int(0.5 * n), len(art)))
    whole_X[:int(0.5 * n), art] = whole_X[:int(0.5 * n), art] + mat

    whole_Y = np.zeros((n, 2))
    whole_Y[:int(0.5 * n), 0] = 1
    whole_Y[int(0.5 * n):, 1] = 1
    t = np.random.permutation(n)
    whole_X = whole_X[t]
    whole_Y = whole_Y[t]
    p0 = len(feature_index)
    whole_X = preprocessing.scale(whole_X)
    whole_X = whole_X[:, feature_index]

    train_Y = whole_Y[:int(0.8 * 0.7 * n)]
    val_Y = whole_Y[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]

    train_X = whole_X[:int(0.8 * 0.7 * n)]
    val_X = whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    new_art = [feature_index.index(a)
               for a in art
               if a in feature_index]
    if deep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)

        Mask_train_X = Mask_whole_X[:int(0.8 * 0.7 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset1 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset1 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset1 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=False)
        val_dataloader1 = DataLoader(val_dataset1, batch_size=batch_size, shuffle=False)
        test_dataloader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)

        feature_num = int(sum(non_constant_features))

        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader1, val_dataloader1, test_dataloader1, \
            train_X.shape[0], val_X.shape[0], test_X.shape[0], new_art, new_art, p0, feature_num, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[
            0], new_art, p0, feature_num
def generate_dataset_4_DL(samples_num, feature_num, batch_size, k, important_feature_num,feature_index,deep_varis):
    whole_X = np.random.uniform(-1, 1, (samples_num, feature_num))
    n = whole_X.shape[0]
    p0 = whole_X.shape[1]

    art = np.array(random.sample(range(p0), important_feature_num))  # the indexes of significant variables

    sign = np.random.choice([-1, 1], len(art))
    beta = np.random.uniform(1, 3, len(art))
    beta = beta * sign
    epsilon = np.random.randn(n)

    wholenew_X = np.zeros((n, len(art)))
    wholenew_X[:, :16] = whole_X[:, art[:16]]
    wholenew_X[:, 16:32] = np.sin(whole_X[:, art[16:32]])
    wholenew_X[:, 32:48] = np.exp(whole_X[:, art[32:48]])
    wholenew_X[:, 48:] = np.maximum(whole_X[:, art[48:]], np.zeros((n, 16)))

    sign2 = np.random.choice([-1, 1], 4)
    beta2 = np.random.uniform(1, 3, 4)
    beta2 = beta2 * sign2

    wholenew_X2 = np.zeros((n, 4))
    wholenew_X2[:, 0] = whole_X[:, art[14]] * whole_X[:, art[15]]
    wholenew_X2[:, 1] = whole_X[:, art[30]] * whole_X[:, art[31]]
    wholenew_X2[:, 2] = whole_X[:, art[46]] * whole_X[:, art[47]]
    wholenew_X2[:, 3] = whole_X[:, art[62]] * whole_X[:, art[63]]


    whole_Y = np.dot(wholenew_X, beta) + np.dot(wholenew_X2, beta2) + epsilon
    whole_Y = np.reshape(whole_Y, (n, 1))

    train_Y = whole_Y[:int(0.8 * 0.7 * n)]
    val_Y = whole_Y[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]
    p0 = len(feature_index)
    whole_X = preprocessing.scale(whole_X)
    whole_X=whole_X[:,feature_index]
    train_X = whole_X[:int(0.8 * 0.7 * n)]
    val_X = whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    new_art = [feature_index.index(a)
               for a in art
               if a in feature_index]
    if deep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)


        Mask_train_X = Mask_whole_X[:int(0.8 * 0.7 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset4 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset4 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset4 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader4 = DataLoader(train_dataset4, batch_size=batch_size, shuffle=True)
        val_dataloader4 = DataLoader(val_dataset4, batch_size=batch_size, shuffle=True)
        test_dataloader4 = DataLoader(test_dataset4, batch_size=batch_size, shuffle=True)
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader4, val_dataloader4, test_dataloader4, \
            train_X.shape[0], val_X.shape[0], test_X.shape[0], new_art, new_art, p0, p0, non_constant_features

    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[
            0], new_art, p0, feature_num

def generate_dataset_6_DL(samples_num, feature_num, batch_size, k, important_feature_num,feature_index,deep_varis):
    whole_X = np.random.uniform(-1, 1, (samples_num, feature_num))
    n = whole_X.shape[0]
    p0 = whole_X.shape[1]

    art = np.array(random.sample(range(p0), important_feature_num))  # the indexes of significant variables

    sign = np.random.choice([-1, 1], len(art))
    beta = np.random.uniform(1, 3, len(art))
    beta = beta * sign
    epsilon = np.random.randn(n)

    wholenew_X = np.zeros((n, len(art)))
    wholenew_X[:, :16] = whole_X[:, art[:16]]
    wholenew_X[:, 16:32] = 2*np.sin(whole_X[:, art[16:32]])
    wholenew_X[:, 32:48] = 2*np.exp(whole_X[:, art[32:48]])
    wholenew_X[:, 48:] = 2*np.maximum(whole_X[:, art[48:]], np.zeros((n, 16)))

    sign2 = np.random.choice([-1, 1], 4)
    beta2 = np.random.uniform(1, 3, 4)
    beta2 = beta2 * sign2

    wholenew_X2 = np.zeros((n, 4))
    wholenew_X2[:, 0] = whole_X[:, art[14]] * whole_X[:, art[15]]
    wholenew_X2[:, 1] = whole_X[:, art[30]] * whole_X[:, art[31]]
    wholenew_X2[:, 2] = whole_X[:, art[46]] * whole_X[:, art[47]]
    wholenew_X2[:, 3] = whole_X[:, art[62]] * whole_X[:, art[63]]


    whole_Y = np.dot(wholenew_X, beta) + np.dot(wholenew_X2, beta2) + epsilon
    whole_Y = np.reshape(whole_Y, (n, 1))

    train_Y = whole_Y[:int(0.8 * 0.7 * n)]
    val_Y = whole_Y[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_Y = whole_Y[int(0.8 * n):]
    p0 = len(feature_index)
    whole_X = preprocessing.scale(whole_X)
    whole_X=whole_X[:,feature_index]
    train_X = whole_X[:int(0.8 * 0.7 * n)]
    val_X = whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
    test_X = whole_X[int(0.8 * n):]
    new_art = [feature_index.index(a)
               for a in art
               if a in feature_index]

    if deep_varis:
        Mask_whole_X, non_constant_features = to_Onehot_matrix(whole_X, k)


        Mask_train_X = Mask_whole_X[:int(0.8 * 0.7 * n)]
        Mask_val_X = Mask_whole_X[int(0.8 * 0.7 * n):int(0.8 * n)]
        Mask_test_X = Mask_whole_X[int(0.8 * n):]

        train_dataset4 = TensorDataset(torch.tensor(Mask_train_X).unsqueeze(1), torch.tensor(train_Y).squeeze(-1))
        val_dataset4 = TensorDataset(torch.tensor(Mask_val_X).unsqueeze(1), torch.tensor(val_Y).squeeze(-1))
        test_dataset4 = TensorDataset(torch.tensor(Mask_test_X).unsqueeze(1), torch.tensor(test_Y).squeeze(-1))

        train_dataloader4 = DataLoader(train_dataset4, batch_size=batch_size, shuffle=True)
        val_dataloader4 = DataLoader(val_dataset4, batch_size=batch_size, shuffle=True)
        test_dataloader4 = DataLoader(test_dataset4, batch_size=batch_size, shuffle=True)


        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_dataloader4, val_dataloader4, test_dataloader4, \
        train_X.shape[0], val_X.shape[0], test_X.shape[0], new_art, new_art, p0, p0, non_constant_features
    else:
        return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X.shape[0], val_X.shape[0], test_X.shape[
            0], new_art, p0, feature_num