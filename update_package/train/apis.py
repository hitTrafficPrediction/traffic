import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # print(A_hat.dtype)
        lfs = lfs.float()
        # print(A_hat.dtype)
        # print(lfs.dtype)
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features, num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    """
    input feature: flow, Speed, hour_minutes_sin, hour_minutes_sin
    return: flow or Speed
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        # 选择流量
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)).float(), \
           torch.from_numpy(np.array(target)).float()


def generate_dataset_speed(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features, num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    """
    input feature: flow, Speed, hour_minutes_sin, hour_minutes_sin
    return: flow or Speed
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        # 选择速度
        target.append(X[:, 1, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)).float(), \
           torch.from_numpy(np.array(target)).float()


def generate_dataset_speed_smooth(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features, num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    """
    input feature: flow, Speed, hour_minutes_sin, hour_minutes_sin
    return: flow or Speed
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        # 选择速度
        target.append(X[:, 1, i + num_timesteps_input: j])

    feature_s = np.array(features)
    target_s = np.array(target)
    for k in range(feature_s.shape[0]):
        feature_s[k, 0, :, 1] = savgol_filter(feature_s[k, 0, :, 1], 19, 3, mode='nearest')
        target_s[k, 0, :] = savgol_filter(target_s[k, 0, :], 19, 3, mode='nearest')

    return torch.from_numpy(feature_s).float(), \
           torch.from_numpy(target_s).float()


class mape_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        mask = torch.absolute(y) <= 1e-3
        result = 100 * (x - y) / y
        result = result.masked_fill(mask, value=0)
        return torch.mean(torch.absolute(result))


def load_data_15min(name):
    #   0,     1,          2,          3,                       4,               5,         6,              7           8                   9
    # index,section_id,direction,point_time,            traffic_flow_total,avg_speed_car,time_calc,     if_next_time, hour_minutes_sin, hour_minutes_cos
    # 1,        G1-1,       上行, 2021-01-25 18:45:00,      324,                100,      20210125184500,     0/1
    # if_next_time值为1表示改行与下一行之间没有空缺值
    df = pd.read_csv(name)
    X = np.array(df[['traffic_flow_total', 'avg_speed_car', 'hour_minutes_sin', 'hour_minutes_cos']]).transpose()
    # X = np.array(df[['traffic_flow_total', 'hour_minutes_sin', 'hour_minutes_cos']]).transpose()
    X = X[np.newaxis, :]

    return X


def load_data_hour(name):
    #   0,     1,          2,          3,                       4,               5,         6,              7           8                   9
    # index,section_id,direction,point_time,            traffic_flow_total,avg_speed_car,time_calc,     if_next_time, hour_minutes_sin, hour_minutes_cos
    # 1,        G1-1,       上行, 2021-01-25 18:45:00,      324,                100,      20210125184500,     0/1
    # if_next_time值为1表示改行与下一行之间没有空缺值
    df = pd.read_csv('D:/algo/龙江交投公路数据/按路段处理/hour_output/' + name)
    X = np.array(df[['traffic_flow_total', 'avg_speed_car', 'hour_sin', 'hour_cos']]).transpose()
    # X = np.array(df[['traffic_flow_total', 'hour_minutes_sin', 'hour_minutes_cos']]).transpose()
    X = X[np.newaxis, :]

    return X


def split_dataset(inputs, target):
    indices = int(0.8 * inputs.shape[0])
    train_inputs = inputs[0:indices, :, :, :]
    train_target = target[0:indices, :, :]
    test_inputs = inputs[indices:, :, :, :]
    test_target = target[indices:, :, :]
    return train_inputs, train_target, test_inputs, test_target


if __name__ == '__main__':
    print()
    X = load_data_15min('G1012-3_up.csv')
    inputs, target = generate_dataset_speed_smooth(X, 96, 96)
    train_inputs, train_target, test_inputs, test_target = split_dataset(inputs, target)

    print(inputs.shape, target.shape)

    x = inputs[0, 0, :, 1]
    y = target[0, 0, :]

    plt.plot(x, 'b')
    plt.plot(y, 'r')
    plt.show()
