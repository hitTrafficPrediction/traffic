import json
import math
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pandas.core.frame import DataFrame
import datetime
import pymysql
from utils.db_operator import get15min_data
from scipy import optimize as op
from fbprophet import Prophet


# 神经网络定义类
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


# 神经网络定义类
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


# 神经网络定义类
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


# prophet 预测模型
def prophet_model_flow(data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        'ds': data.time,
        'y': data.flow,
    })

    df['cap'] = data.flow.values.max()
    df['floor'] = data.flow.values.min()

    # 考虑日周期性与周周期性
    m = Prophet(
        changepoint_prior_scale=0.05,
        daily_seasonality=True,
        weekly_seasonality=True,  # 周周期性
        yearly_seasonality=False,  # 年周期性
        growth="logistic",
    )

    m.fit(df)

    # 直接预测未来14天的数值
    future = m.make_future_dataframe(periods=14 * 24, freq='H')  # 预测时长
    future['cap'] = data.flow.values.max()
    future['floor'] = data.flow.values.min()

    forecast = m.predict(future)

    # fig = m.plot_components(forecast)
    # fig1 = m.plot(forecast)

    return forecast


# prophet 预测模型
def prophet_model_speed(data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        'ds': data.time,
        'y': data.speed,
    })

    df['cap'] = data.speed.values.max()
    df['floor'] = data.speed.values.min()

    # 考虑日周期性与周周期性
    m = Prophet(
        changepoint_prior_scale=0.05,
        daily_seasonality=True,
        weekly_seasonality=True,  # 周周期性
        yearly_seasonality=False,  # 年周期性
        growth="logistic",
    )

    m.fit(df)

    # 直接预测未来14天的数值
    future = m.make_future_dataframe(periods=14 * 24, freq='H')  # 预测时长
    future['cap'] = data.speed.values.max()
    future['floor'] = data.speed.values.min()

    forecast = m.predict(future)

    # fig = m.plot_components(forecast)
    # fig1 = m.plot(forecast)

    return forecast


# 依据显示时间计算对应时间标签
def time_sin_minutes(index):
    # 2021-01-25 18:45:00
    date, hours_minutes = index.split(' ')
    hours, minutes, second = hours_minutes.split(':')
    index = float(hours + minutes) * np.pi / 2400
    # print(index)
    return np.sin(index)


# 依据显示时间计算对应时间标签
def time_cos_minutes(index):
    # 2021-01-25 18:45:00
    date, hours_minutes = index.split(' ')
    hours, minutes, second = hours_minutes.split(':')
    index = float(hours + minutes) * np.pi / 2400
    # print(index)
    return np.cos(index)


# 依据显示时间计算对应时间标签
def time_sin_hour(index):
    # 2021-01-25 18
    date, hours = index.split(' ')
    index = float(hours) * np.pi / 24
    # print(index)
    return np.sin(index)


# 依据显示时间计算对应时间标签
def time_cos_hour(index):
    # 2021-01-25 18
    date, hours = index.split(' ')
    index = float(hours) * np.pi / 24
    # print(index)
    return np.cos(index)


def days_interval(start, end):
    """
    计算两天时间间隔
    """
    date1 = datetime.datetime.strptime(end[0:10], "%Y-%m-%d")
    date2 = datetime.datetime.strptime(start[0:10], "%Y-%m-%d")
    num = (date1 - date2).days
    return num + 1


def get15min_data_week(section_id, custom_time):
    result_up = []
    result_down = []

    custom_time = next_day(custom_time, -7)
    for i in range(7):
        data_up, data_down = get15min_data(section_id, custom_time)
        for item_up in data_up:
            result_up.append(item_up)
        for item_down in data_down:
            result_down.append(item_down)

        custom_time = next_day(custom_time, 1)

    return result_up, result_down


def read_local_data_week(section_id):
    """
    本地测试用, 读取一个星期的历史数据
    : return input_data_up, input_data_down 历史数据

    traffic_flow_total, avg_speed_car, point_time
    shape (96*7) list of tuple
    """
    local_dir = 'example_data/'
    time_step = 96 * 7

    result_up = []
    result_down = []

    #   0,     1,          2,          3,            4,               5,         6,        7           8                   9
    # index,section_id,direction,point_time,traffic_flow_total,avg_speed_car,time_calc,if_next_time, hour_minutes_sin, hour_minutes_cos
    df_up = pd.read_csv(local_dir + section_id + '_up.csv')
    df_down = pd.read_csv(local_dir + section_id + '_down.csv')
    data_up = df_up[['traffic_flow_total', 'avg_speed_car', 'point_time']]
    data_down = df_down[['traffic_flow_total', 'avg_speed_car', 'point_time']]

    max_number = min(data_up.shape[0], data_down.shape[0])
    indices = int(np.random.uniform(low=0.0, high=max_number - time_step))
    for i in range(time_step):
        result_up.append((data_up.iloc[indices + i, 0], data_up.iloc[indices + i, 1], data_up.iloc[indices + i, 2]))
        result_down.append(
            (data_down.iloc[indices + i, 0], data_down.iloc[indices + i, 1], data_down.iloc[indices + i, 2]))

    return result_up, result_down


def read_local_data(section_id, mode):
    """
    本地测试用
    : return input_data_up, input_data_down 历史数据

    traffic_flow_total, avg_speed_car, point_time
    shape (96) list of tuple
    """
    if 'minute' in mode:
        local_dir = 'example_data/'
        time_step = 96
    else:
        local_dir = 'example_data/hour_output/'
        time_step = 24
    result_up = []
    result_down = []

    #   0,     1,          2,          3,            4,               5,         6,        7           8                   9
    # index,section_id,direction,point_time,traffic_flow_total,avg_speed_car,time_calc,if_next_time, hour_minutes_sin, hour_minutes_cos
    df_up = pd.read_csv(local_dir + section_id + '_up.csv')
    df_down = pd.read_csv(local_dir + section_id + '_down.csv')
    data_up = df_up[['traffic_flow_total', 'avg_speed_car', 'point_time']]
    data_down = df_down[['traffic_flow_total', 'avg_speed_car', 'point_time']]

    max_number = min(data_up.shape[0], data_down.shape[0])
    indices = int(np.random.uniform(low=0.0, high=max_number - time_step))
    for i in range(time_step):
        result_up.append((data_up.iloc[indices + i, 0], data_up.iloc[indices + i, 1], data_up.iloc[indices + i, 2]))
        result_down.append(
            (data_down.iloc[indices + i, 0], data_down.iloc[indices + i, 1], data_down.iloc[indices + i, 2]))

    return result_up, result_down


def show_img(result, mode):
    """
    本地测试用
    绘制输入与输出的对比图
    """
    if mode == 'speed':
        dim = 1
    elif mode == 'flow':
        dim = 0
    elif mode == 'index':
        dim = 2
    else:
        print('mode error')
        sys.exit(0)
    result = np.array(result)
    plt.plot(result[:, dim])
    plt.show()


def traffic_index(flow, speed, mode, name):
    """
    计算拥堵指数
    默认设计时速为 120km/h
    : param mode 用于确定时间粒度 -> granularity
    : param name 用于确定车道数 -> lanes

    : return 拥堵指数 int
    """
    # 设置车道数，默认G1京哈高速拥有单向4车道，其他高速拥有单向2车道
    if 'G1' in name:
        lanes = 4
    else:
        lanes = 2
    # 设置粒度系数，即1小时内拥有多少单位的’粒度‘，便于流量转换
    if 'minutes' in mode:
        granularity = 4
    else:
        granularity = 1

    # 设置车头时距
    t = 3.15

    # 设置车道数系数
    if lanes == 2:
        k = 1 + 0.85
    else:
        # lanes = 4
        k = 1 + 0.85 + 0.7 + 0.55

    # 计算参数
    # c = 0.8×（3600/t ×0.98×3.75×0.97）×0.95×（1+0.85）
    # c = 0.8×（3600/t ×0.98×3.75×0.97）×0.95×（1+0.85+0.7+0.55）
    c = 0.8 * (3600 * 0.98 * 3.75 * 0.97) * 0.95 * k / t

    vc = flow * granularity / c

    # 判断拥堵指数，具体判据请参考说明文档
    # 更新：
    # 拥堵指数划分为0-10等级，1级最不拥堵，10级最拥堵
    if 0 <= vc <= 0.11 or 116 <= speed:
        return 1
    elif 0.11 < vc <= 0.23 or 113 <= speed < 116:
        return 2
    elif 0.23 < vc <= 0.34 or 109 <= speed < 113:
        return 3
    elif 0.34 < vc <= 0.47 or 103 <= speed < 109:
        return 4
    elif 0.47 < vc <= 0.61 or 96 <= speed < 103:
        return 5
    elif 0.61 < vc <= 0.74 or 90 <= speed < 96:
        return 6
    elif 0.74 < vc <= 0.81 or 84 <= speed < 90:
        return 7
    elif 0.81 < vc <= 0.88 or 78 <= speed < 84:
        return 8
    elif 0.88 < vc <= 0.91 or 63 <= speed < 78:
        return 9
    elif 0.94 < vc or speed < 63:
        return 10

    else:
        print('拥堵指数计算错误 params "flow, speed, mode, name"')
        print(flow, speed, mode, name)
        sys.exit(1)


def data_Pre_process(data_up, data_down):
    """
    数据预处理
    : param input_data_xx columns for
    traffic_flow_total, avg_speed_car, point_time
    shape (time_step) type list

    point_time like 2021-01-25 18:45:00; time_step = 96

    : return input_xx columns(features) for
    traffic_flow_total, avg_speed_car, sin(point_time), cos(point_time)
    shape (1, 1, 96, 4) type numpy.array
    """
    # 将list转化为DataFrame, 便于操作
    data_up = DataFrame(data_up)
    data_down = DataFrame(data_down)

    input_up = []
    input_down = []
    for i in range(data_up.shape[0]):
        input_up.append(
            (data_up.iloc[i, 0], data_up.iloc[i, 1], time_sin_minutes(data_up.iloc[i, 2]),
             time_cos_minutes(data_up.iloc[i, 2])))
        input_down.append(
            (data_down.iloc[i, 0], data_down.iloc[i, 1], time_sin_minutes(data_down.iloc[i, 2]),
             time_cos_minutes(data_down.iloc[i, 2])))

    # 数据整形
    input_up = np.array(input_up)
    input_down = np.array(input_down)
    input_up = input_up[np.newaxis, np.newaxis, :]
    input_down = input_down[np.newaxis, np.newaxis, :]
    return torch.from_numpy(np.array(input_up)).float(), \
           torch.from_numpy(np.array(input_down)).float()


def data_Fin_process(flow_result_up, flow_result_down, speed_result_up, speed_result_down, mode, name):
    """
    整合预测数据, 及计算拥堵指数
    : param xx_result_xx columns is flow/speed; numpy.array
    shape (96)
    : param name 用于确定车道数量以计算拥堵指数

    : return result_up, result_down; list
    result_xx columns are
    traffic_flow_total, avg_speed_car, traffic_index
    shape (96, 3)或(24,3)
    """
    result_up = []
    result_down = []
    if mode == '15minutes':
        for i in range(96):
            result_up.append(
                (flow_result_up[i], speed_result_up[i],
                 traffic_index(flow_result_up[i], speed_result_up[i], mode, name)))
            result_down.append(
                (flow_result_down[i], speed_result_down[i],
                 traffic_index(flow_result_down[i], speed_result_down[i], mode, name)))
    elif mode == 'hour':
        for i in range(24):
            flow_up = (flow_result_up[4 * i] + flow_result_up[4 * i + 1] + flow_result_up[4 * i + 2] + flow_result_up[
                4 * i + 3])
            speed_up = (speed_result_up[4 * i] + speed_result_up[4 * i + 1] + speed_result_up[4 * i + 2] +
                        speed_result_up[4 * i + 3]) / 4
            flow_down = (flow_result_down[4 * i] + flow_result_down[4 * i + 1] + flow_result_down[4 * i + 2] +
                         flow_result_down[4 * i + 3])
            speed_down = (speed_result_down[4 * i] + speed_result_down[4 * i + 1] + speed_result_down[4 * i + 2] +
                          speed_result_down[4 * i + 3]) / 4
            result_up.append((flow_up, speed_up, traffic_index(flow_up, speed_up, mode, name)))
            result_down.append((flow_down, speed_down, traffic_index(flow_down, speed_down, mode, name)))
    else:
        print('mode error')
        sys.exit(0)

    return result_up, result_down


def make_prediction(name, device, mode, data_up, data_down):
    """
    :参数 name 对应 section_id
    :参数 device 模型运行设备
    :参数 mode 模型运行模式, 15minutes/hour
    :参数 data_xx 对应 上行、下行历史数据 list
    """
    # 模型参数定义
    time_step = 96

    # 模型初始化
    A_wave = torch.FloatTensor([[1]]).to(device=device)
    net = STGCN(A_wave.shape[0], 4, time_step, time_step).to(device=device)

    # 数据预处理
    # traffic_flow_total, avg_speed_car, sin(point_time), cos(point_time)
    # shape(1, 1, time_step, 4)
    input_up, input_down = data_Pre_process(data_up, data_down)

    with torch.no_grad():
        net.eval()

        # 预测流量（上行up、下行down） config/15minutes/section_id_up
        state_dist = torch.load('config/15minutes/' + name + '_up', map_location=torch.device(device))
        net.load_state_dict(state_dist['state_dist'])
        # result_xx from NET shape (1, 1, time_step)
        flow_result_up = net(A_wave, input_up)
        flow_result_up = flow_result_up.cpu().numpy().astype(np.int)
        flow_result_up = np.squeeze(flow_result_up)
        # result_xx now shape (time_step)
        flow_result_up[flow_result_up < 0] = 0

        state_dist = torch.load('config/15minutes/' + name + '_down', map_location=torch.device(device))
        net.load_state_dict(state_dist['state_dist'])
        # result_xx from NET shape (1, 1, time_step)
        flow_result_down = net(A_wave, input_down)
        flow_result_down = flow_result_down.cpu().numpy().astype(np.int)
        flow_result_down = np.squeeze(flow_result_down)
        # result_xx now shape (time_step)
        flow_result_down[flow_result_down < 0] = 0

        # 修改
        # 预测速度（上行up、下行down） config/15minutes/section_id_up
        state_dist = torch.load('config/speed', map_location=torch.device(device))
        net.load_state_dict(state_dist)
        # result_xx from NET shape (1, 1, time_step)
        speed_result_up = net(A_wave, input_up)
        speed_result_up = speed_result_up.cpu().numpy().astype(np.int)
        speed_result_up = np.squeeze(speed_result_up)
        # result_xx now shape (time_step)
        speed_result_up[speed_result_up < 60] = 60
        # speed_result_up[speed_result_up > 120] = 120

        # result_xx from NET shape (1, 1, time_step)
        speed_result_down = net(A_wave, input_down)
        speed_result_down = speed_result_down.cpu().numpy().astype(np.int)
        speed_result_down = np.squeeze(speed_result_down)
        # result_xx now shape (time_step)
        speed_result_down[speed_result_down < 60] = 60
        # speed_result_up[speed_result_up > 120] = 120

    # 依据输入数据速度的平均值确定预测的速度值
    # input_up = input_up.cpu().numpy()
    # input_down = input_down.cpu().numpy()
    # avg_speed_up = np.full(96, np.mean(input_up[0, 0, :, 1]))
    # avg_speed_down = np.full(96, np.mean(input_down[0, 0, :, 1]))

    # 整合预测数据, 及计算拥堵指数
    # return result_up, result_down
    # result_xx columns are
    # traffic_flow_total, avg_speed_car, traffic_index
    # shape (96, 3)
    # type list
    # return data_Fin_process(flow_result_up, flow_result_down, avg_speed_up, avg_speed_down, mode, name)
    return data_Fin_process(flow_result_up, flow_result_down, speed_result_up, speed_result_down, mode, name)


def make_prediction_iteration(name, device, mode, input_up, input_down):
    """
    :参数 name 对应 section_id
    :参数 device 模型运行设备
    :参数 mode 模型运行模式, 15minutes/hour
    :参数 input_xx 对应 上行、下行历史数据, 输入与输出格式一致, 便于迭代运行
    shape(1, 1, time_step, 4) tensor
    """
    # 模型参数定义
    time_step = 96

    # 模型初始化
    A_wave = torch.FloatTensor([[1]]).to(device=device)
    net = STGCN(A_wave.shape[0], 4, time_step, time_step).to(device=device)

    # 数据预处理
    # traffic_flow_total, avg_speed_car, sin(point_time), cos(point_time)
    # shape(1, 1, time_step, 4)
    # input_up, input_down = data_Pre_process(data_up, data_down)
    with torch.no_grad():
        net.eval()

        # 预测流量（上行up、下行down） config/15minutes/section_id_up
        state_dist = torch.load('config/15minutes/' + name + '_up', map_location=torch.device(device))
        net.load_state_dict(state_dist['state_dist'])
        # result_xx from NET shape (1, 1, time_step)
        flow_result_up = net(A_wave, input_up)
        flow_result_up = flow_result_up.cpu().numpy().astype(np.int)
        flow_result_up = np.squeeze(flow_result_up)
        # result_xx now shape (time_step)
        # 设定阈值
        flow_result_up[flow_result_up < 0] = 0
        flow_result_up[flow_result_up > 375] = 375

        state_dist = torch.load('config/15minutes/' + name + '_down', map_location=torch.device(device))
        net.load_state_dict(state_dist['state_dist'])
        # result_xx from NET shape (1, 1, time_step)
        flow_result_down = net(A_wave, input_down)
        flow_result_down = flow_result_down.cpu().numpy().astype(np.int)
        flow_result_down = np.squeeze(flow_result_down)
        # result_xx now shape (time_step)
        # 设定阈值
        flow_result_down[flow_result_down < 0] = 0
        flow_result_down[flow_result_down > 375] = 375

        # 修改
        # 预测速度（上行up、下行down） config/15minutes/section_id_up
        state_dist = torch.load('config/speed', map_location=torch.device(device))
        net.load_state_dict(state_dist)
        # result_xx from NET shape (1, 1, time_step)
        speed_result_up = net(A_wave, input_up)
        speed_result_up = speed_result_up.cpu().numpy().astype(np.int)
        speed_result_up = np.squeeze(speed_result_up)
        # result_xx now shape (time_step)
        # 设定阈值
        speed_result_up[speed_result_up < 60] = 60
        speed_result_up[speed_result_up > 120] = 120

        # result_xx from NET shape (1, 1, time_step)
        speed_result_down = net(A_wave, input_down)
        speed_result_down = speed_result_down.cpu().numpy().astype(np.int)
        speed_result_down = np.squeeze(speed_result_down)
        # result_xx now shape (time_step)
        # 设定阈值
        speed_result_down[speed_result_down < 60] = 60
        speed_result_down[speed_result_down > 120] = 120

    # (96,) to (1, 1, 96, 4)
    # 迭代数据返回
    # next_data_up = np.concatenate((flow_result_up[np.newaxis, np.newaxis, :, np.newaxis], input_up[:, :, :, 1:2]), axis=3)
    next_data_up = np.concatenate(
        (flow_result_up[np.newaxis, np.newaxis, :, np.newaxis], speed_result_up[np.newaxis, np.newaxis, :, np.newaxis]),
        axis=3)
    next_data_up = np.concatenate((next_data_up, input_up[:, :, :, :-2]), axis=3)

    # next_data_down = np.concatenate((flow_result_down[np.newaxis, np.newaxis, :, np.newaxis], input_down[:, :, :, 1:2]), axis=3)
    next_data_down = np.concatenate((flow_result_down[np.newaxis, np.newaxis, :, np.newaxis],
                                     speed_result_down[np.newaxis, np.newaxis, :, np.newaxis]), axis=3)
    next_data_down = np.concatenate((next_data_down, input_down[:, :, :, :-2]), axis=3)
    # print(next_data_down.shape)

    # 整合预测数据, 及计算拥堵指数
    # return result_up, result_down
    # result_xx columns are
    # traffic_flow_total, avg_speed_car, traffic_index
    # shape (96, 3)
    # type list
    # return data_Fin_process(flow_result_up, flow_result_down, avg_speed_up, avg_speed_down, mode, name)
    result_up, result_down = data_Fin_process(flow_result_up, flow_result_down, speed_result_up, speed_result_down,
                                              mode, name)
    return result_up, result_down, \
           torch.from_numpy(next_data_up).float(), torch.from_numpy(next_data_down).float()


def prophet_pre_process(data_up, data_down):
    """
    将输入数据处理为pandas表格
    """
    data_up_pd = pd.DataFrame(columns=['flow', 'speed', 'time'])
    data_down_pd = pd.DataFrame(columns=['flow', 'speed', 'time'])
    for i in data_up:
        data_up_pd = data_up_pd.append({'flow': i[0], 'speed': i[1], 'time': i[2]}, ignore_index=True)
    for j in data_down:
        data_down_pd = data_down_pd.append({'flow': j[0], 'speed': j[1], 'time': j[2]}, ignore_index=True)

    # data_up_pd['time'] = pd.to_datetime(data_up_pd['time'], format='%Y-%m-%d %H:%M:%S')
    # data_down_pd['time'] = pd.to_datetime(data_down_pd['time'], format='%Y-%m-%d %H:%M:%S')

    return data_up_pd, data_down_pd


def prophet_fin_process(flow_up, flow_down, speed_up, speed_down, name):
    """
    整合预测数据, 及计算拥堵指数
    : param xx_xx is flow/speed; list
    len (24*14)
    : param name 用于确定车道数量以计算拥堵指数

    : return result_up, result_down; list
    result_xx columns are
    traffic_flow_total, avg_speed_car, traffic_index
    shape (14, 24, 3)
    14天的24小时的 流量 速度 拥堵指数
    """
    result_up = []
    result_down = []

    # 14天中的24小时
    for i in range(14):
        day_temp_up = []
        day_temp_down = []
        for j in range(24):
            day_temp_up.append((flow_up[i * 24 + j], speed_up[i * 24 + j],
                                traffic_index(flow_up[i * 24 + j], speed_up[i * 24 + j], 'hour', name)))
            day_temp_down.append((flow_down[i * 24 + j], speed_down[i * 24 + j],
                                  traffic_index(flow_down[i * 24 + j], speed_down[i * 24 + j], 'hour', name)))

        result_up.append(day_temp_up)
        result_down.append(day_temp_down)

    return result_up, result_down


def make_prediction_prophet(name, data_up, data_down):
    """
    一次预测未来14天数据

    :参数 name 对应 section_id
    :参数 device 模型运行设备
    :参数 mode 模型运行模式, 15minutes/hour
    :参数 data_xx 对应 上行、下行历史数据 list
    """
    # 数据预处理
    # columns=['flow', 'speed', 'time']
    # 一个星期
    # pandas 96*7 rows
    input_up, input_down = prophet_pre_process(data_up, data_down)

    # input_up = pd.read_csv('example_data/up.csv').sample(frac=0.2)
    # input_down = pd.read_csv('example_data/down.csv').sample(frac=0.2)
    # input_up = pd.read_csv('example_data/up.csv')
    # input_down = pd.read_csv('example_data/down.csv')

    # 预测流量
    flow_up = prophet_model_flow(input_up)
    flow_down = prophet_model_flow(input_down)

    # 预测速度
    speed_up = prophet_model_speed(input_up)
    speed_down = prophet_model_speed(input_down)

    # numpy.ndarray 24*14
    flow_result_up = flow_up.yhat.values[-24 * 14:]
    flow_result_down = flow_down.yhat.values[-24 * 14:]
    speed_result_up = speed_up.yhat.values[-24 * 14:]
    speed_result_down = speed_down.yhat.values[-24 * 14:]

    # 整合预测数据, 及计算拥堵指数
    # : return result_up, result_down;
    # list result_xx columns are
    # traffic_flow_total, avg_speed_car, traffic_index
    # shape(14, 24, 3)
    # 14天的24小时的 流量 速度 拥堵指数
    return prophet_fin_process(flow_result_up, flow_result_down, speed_result_up, speed_result_down, name)


def day_data_process(result_up, result_down, days, name):
    """
    根据单日的预测结果，推算多日的交通情况
    result_up, result_down; list
    result_xx columns are
    traffic_flow_total, avg_speed_car, traffic_index
    shape (24, 3), hour粒度
    峰值时间 10, 11, 12, 14, 15, 16, 17
    平值时间 7, 8, 9, 13, 18, 19, 20
    谷值时间 1, 2, 3, 4, 5, 6, 21, 22, 23, 24

    : return
    columns are
    peak_value, normal_value, valley_value tuples for flow, speed, and index
    example (processed_xx[0])
    ( (77.38573605907594, 90.88928731770665, 4),      peak -> flow,speed,index(拥堵指数）
      (74.39937376149416, 90.88928731770665, 4),      normal -> flow,speed,index(拥堵指数）
      (70.71186553317577, 90.88928731770665, 4)   )   valley -> flow,speed,index(拥堵指数）
    len(processed_xx) = prediction_days
    """
    processed_up = []
    processed_down = []
    peak_time = [9, 10, 11, 13, 14, 15, 16]
    normal_time = [6, 7, 8, 12, 17, 18, 19]
    valley_time = [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]

    data_up = np.array(result_up)
    data_down = np.array(result_down)

    flow_peak_up_sum = np.sum(data_up[peak_time, 0])
    speed_peak_up_avg = np.mean(data_up[peak_time, 1])
    index_peak_up = traffic_index(flow_peak_up_sum / len(peak_time), speed_peak_up_avg, 'hour', name)
    flow_normal_up_sum = np.mean(data_up[normal_time, 0])
    speed_normal_up_avg = np.mean(data_up[normal_time, 1])
    index_normal_up = traffic_index(flow_normal_up_sum / len(normal_time), speed_normal_up_avg, 'hour', name)
    flow_valley_up_sum = np.mean(data_up[valley_time, 0])
    speed_valley_up_avg = np.mean(data_up[valley_time, 1])
    index_valley_up = traffic_index(flow_valley_up_sum / len(valley_time), speed_valley_up_avg, 'hour', name)

    flow_peak_down_sum = np.mean(data_down[peak_time, 0])
    speed_peak_down_avg = np.mean(data_down[peak_time, 1])
    index_peak_down = traffic_index(flow_peak_down_sum / len(peak_time), speed_peak_down_avg, 'hour', name)
    flow_normal_down_sum = np.mean(data_down[normal_time, 0])
    speed_normal_down_avg = np.mean(data_down[normal_time, 1])
    index_normal_down = traffic_index(flow_normal_down_sum / len(normal_time), speed_normal_down_avg, 'hour',
                                      name)
    flow_valley_down_sum = np.mean(data_down[valley_time, 0])
    speed_valley_down_avg = np.mean(data_down[valley_time, 1])
    index_valley_down = traffic_index(flow_valley_down_sum / len(valley_time), speed_valley_down_avg, 'hour',
                                      name)

    # 基于一天的预测结果，拓展预测天数
    for i in range(days):
        random_factor = np.random.uniform(0.9, 1.1)
        processed_up.append(((flow_peak_up_sum * random_factor, speed_peak_up_avg * random_factor, index_peak_up),
                             (flow_normal_up_sum * random_factor, speed_normal_up_avg * random_factor, index_normal_up),
                             (
                                 flow_valley_up_sum * random_factor, speed_valley_up_avg * random_factor,
                                 index_valley_up)))
        processed_down.append(
            ((flow_peak_down_sum * random_factor, speed_peak_down_avg * random_factor, index_peak_down),
             (flow_normal_down_sum * random_factor, speed_normal_down_avg * random_factor, index_normal_down),
             (flow_valley_down_sum * random_factor, speed_valley_down_avg * random_factor, index_valley_down)))

    # : processed_xx columns are
    # peak_value, normal_value, valley_value tuples for flow, speed, and index
    # example (processed_xx[0])
    # ( (77.38573605907594, 90.88928731770665, 4),      peak -> flow,speed,index(拥堵指数）
    #   (74.39937376149416, 90.88928731770665, 4),      normal -> flow,speed,index(拥堵指数）
    #   (70.71186553317577, 90.88928731770665, 4)   )   valley -> flow,speed,index(拥堵指数）
    # len(processed_xx) = prediction_days
    return processed_up, processed_down


def holiday_data_process(result_up, result_down, days, name):
    """
    根据单日的预测结果，推算节假日的交通情况
    : result_xx columns are
    traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
    example (result_xx[0])
    (69.0, 100.0, 4) -> traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
    shape (96) list of tuple

    :return
    48 粒度 of (traffic_flow_total, avg_speed_car, traffic_index(拥堵指数) ) a day
    example (result_xx[0], which is the first day's data)
    [79.41736302394557, 118.53337764767996, 4.0], 00:30's data for traffic_flow_total, avg_speed_car, traffic_index(拥堵指数)
    [90.08536701223676, 118.53337764767996, 4.0], 01:00's data
    ...
    [99.56803722405115, 118.53337764767996, 4.0], 11:30's data
    [96.01203589462077, 118.53337764767996, 4.0], 24:00's data
    len(processed_xx) ==  prediction_days+6, list of tuple/list
    """

    result_up = DataFrame(result_up)
    result_down = DataFrame(result_down)
    base_up = np.full((48, 3), 0)
    base_down = np.full((48, 3), 0)
    processed_up = []
    process_down = []
    for i in range(48):
        # 流量
        base_up[i, 0] = result_up.iloc[2 * i, 0] + result_up.iloc[2 * i + 1, 0]
        base_down[i, 0] = result_down.iloc[2 * i, 0] + result_down.iloc[2 * i + 1, 0]
        # 速度
        base_up[i, 1] = (result_up.iloc[2 * i, 1] + result_up.iloc[2 * i + 1, 1]) / 2
        base_down[i, 1] = (result_down.iloc[2 * i, 1] + result_down.iloc[2 * i + 1, 1]) / 2

    # before holiday
    for i in range(3):
        low_random_factor = np.random.uniform(1.15, 1.25)
        temp_up = base_up * low_random_factor
        temp_down = base_down * low_random_factor
        # 重新计算拥堵指数
        for j in range(48):
            temp_up[j, 2] = traffic_index(2 * temp_up[j, 0], temp_up[j, 1], 'hour', name)
            temp_down[j, 2] = traffic_index(2 * temp_down[j, 0], temp_down[j, 1], 'hour', name)
        processed_up.append(temp_up.tolist())
        process_down.append(temp_down.tolist())

    # during holiday
    for i in range(days):
        high_random_factor = np.random.uniform(1.25, 1.35)
        temp_up = base_up * high_random_factor
        temp_down = base_down * high_random_factor
        # 重新计算拥堵指数
        for j in range(48):
            temp_up[j, 2] = traffic_index(2 * temp_up[j, 0], temp_up[j, 1], 'hour', name)
            temp_down[j, 2] = traffic_index(2 * temp_down[j, 0], temp_down[j, 1], 'hour', name)
        processed_up.append(temp_up.tolist())
        process_down.append(temp_down.tolist())

    # after holiday
    for i in range(3):
        low_random_factor = np.random.uniform(0.95, 1.05)
        temp_up = base_up * low_random_factor
        temp_down = base_down * low_random_factor
        # 重新计算拥堵指数
        for j in range(48):
            temp_up[j, 2] = traffic_index(2 * temp_up[j, 0], temp_up[j, 1], 'hour', name)
            base_down[j, 2] = traffic_index(2 * temp_down[j, 0], temp_down[j, 1], 'hour', name)
        processed_up.append(temp_up.tolist())
        process_down.append(temp_down.tolist())

    # : processed_xx columns are
    # 48 粒度 of (traffic_flow_total, avg_speed_car, traffic_index(拥堵指数) ) a day
    # example (result_xx[0], which is the first day's data)
    # [79.41736302394557, 118.53337764767996, 4.0], 00:30's data for traffic_flow_total, avg_speed_car, traffic_index(拥堵指数)
    # [90.08536701223676, 118.53337764767996, 4.0], 01:00's data
    # ...
    # [99.56803722405115, 118.53337764767996, 4.0], 11:30's data
    # [96.01203589462077, 118.53337764767996, 4.0], 24:00's data
    # len(processed_xx) ==  prediction_days+6, list of tuple/list
    return processed_up, process_down


def holiday_data_process_hour(result_up, result_down, days, name):
    """
    根据单日的预测结果，推算节假日的交通情况
    : result_xx columns are
    traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
    example (result_xx[0])
    (69.0, 100.0, 4) -> traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
    shape (24) list of tuple

    :return
    24 粒度 of (traffic_flow_total, avg_speed_car, traffic_index(拥堵指数) ) one day
    example (result_xx[0], which is the first day's data)
    [79.41736302394557, 118.53337764767996, 4.0], 00:00's data for traffic_flow_total, avg_speed_car, traffic_index(拥堵指数)
    [90.08536701223676, 118.53337764767996, 4.0], 01:00's data
    ...
    [99.56803722405115, 118.53337764767996, 4.0], 23:00's data
    [96.01203589462077, 118.53337764767996, 4.0], 24:00's data
    len(processed_xx) ==  prediction_days+6, list of tuple/list
    """

    result_up = DataFrame(result_up)
    result_down = DataFrame(result_down)
    base_up = np.array(result_up)
    base_down = np.array(result_down)
    processed_up = []
    process_down = []

    # before holiday
    for i in range(3):
        low_random_factor = np.random.uniform(1.15, 1.25)
        temp_up = base_up * low_random_factor
        temp_down = base_down * low_random_factor
        # 重新计算拥堵指数
        for j in range(24):
            temp_up[j, 2] = traffic_index(temp_up[j, 0], temp_up[j, 1], 'hour', name)
            temp_down[j, 2] = traffic_index(temp_down[j, 0], temp_down[j, 1], 'hour', name)
        processed_up.append(temp_up.tolist())
        process_down.append(temp_down.tolist())

    # during holiday
    for i in range(days):
        high_random_factor = np.random.uniform(1.25, 1.35)
        temp_up = base_up * high_random_factor
        temp_down = base_down * high_random_factor
        # 重新计算拥堵指数
        for j in range(24):
            temp_up[j, 2] = traffic_index(temp_up[j, 0], temp_up[j, 1], 'hour', name)
            temp_down[j, 2] = traffic_index(temp_down[j, 0], temp_down[j, 1], 'hour', name)
        processed_up.append(temp_up.tolist())
        process_down.append(temp_down.tolist())

    # after holiday
    for i in range(3):
        low_random_factor = np.random.uniform(0.95, 1.05)
        temp_up = base_up * low_random_factor
        temp_down = base_down * low_random_factor
        # 重新计算拥堵指数
        for j in range(24):
            temp_up[j, 2] = traffic_index(temp_up[j, 0], temp_up[j, 1], 'hour', name)
            base_down[j, 2] = traffic_index(temp_down[j, 0], temp_down[j, 1], 'hour', name)
        processed_up.append(temp_up.tolist())
        process_down.append(temp_down.tolist())

    # : processed_xx columns are
    # 24 粒度 of (traffic_flow_total, avg_speed_car, traffic_index(拥堵指数) ) one day
    # example (result_xx[0], which is the first day's data)
    # [79.41736302394557, 118.53337764767996, 4.0], 00:00's data for traffic_flow_total, avg_speed_car, traffic_index(拥堵指数)
    # [90.08536701223676, 118.53337764767996, 4.0], 01:00's data
    # ...
    # [99.56803722405115, 118.53337764767996, 4.0], 23:00's data
    # [96.01203589462077, 118.53337764767996, 4.0], 24:00's data
    # len(processed_xx) ==  prediction_days+6, list of tuple/list
    return processed_up, process_down


def congestion_data_process(result_up, result_down, days, name):
    """
    根据单日的预测结果，推算其他日的拥堵指数情况
    : result_xx columns are
    traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
    example (result_xx[0])
    (69.0, 100.0, 4) -> traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
    shape (96) list of tuple

    :return
    columns are
    48 粒度 of traffic_index(拥堵指数)
    example (result_xx[0])
    [1, 1, ... 5, 3]
    """

    result_up = DataFrame(result_up)
    result_down = DataFrame(result_down)
    base_up = np.full((48, 3), 0)
    base_down = np.full((48, 3), 0)
    processed_up = []
    process_down = []
    for i in range(48):
        # 流量
        base_up[i, 0] = result_up.iloc[2 * i, 0] + result_up.iloc[2 * i + 1, 0]
        base_down[i, 0] = result_down.iloc[2 * i, 0] + result_down.iloc[2 * i + 1, 0]
        # 速度
        base_up[i, 1] = (result_up.iloc[2 * i, 1] + result_up.iloc[2 * i + 1, 1]) / 2
        base_down[i, 1] = (result_down.iloc[2 * i, 1] + result_down.iloc[2 * i + 1, 1]) / 2

    for i in range(days):
        high_random_factor = np.random.uniform(0.9, 1.1)
        temp_up = base_up * high_random_factor
        temp_down = base_down * high_random_factor
        # 重新计算拥堵指数
        for j in range(48):
            temp_up[j, 2] = traffic_index(2 * temp_up[j, 0], temp_up[j, 1], 'hour', name)
            temp_down[j, 2] = traffic_index(2 * temp_down[j, 0], temp_down[j, 1], 'hour', name)
        processed_up.append(temp_up[:, 2].tolist())
        process_down.append(temp_down[:, 2].tolist())

    # : processed_xx columns are
    # 48 粒度 of traffic_index(拥堵指数)
    # example (result_xx[0])
    # [1, 1, ... 5, 3]
    return processed_up, process_down


def next_time_minutes(now, count):
    # 计算之前，之后时间对应的时间字符串
    # 2021-01-25 18:45:00
    next_time = (datetime.datetime.strptime(now, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=+count)).strftime(
        '%Y-%m-%d %H:%M:%S')

    return next_time


def next_day(now, count):
    # 计算之前，之后时间对应的时间字符串
    # 2021-01-25
    next_time = (datetime.datetime.strptime(now, '%Y-%m-%d %H') + datetime.timedelta(days=+count)).strftime(
        '%Y-%m-%d %H')

    return next_time


def load_user_config_sql():
    """
    从配置文件中获得数据库信息
    """
    try:
        with open("./config/db_config.json") as f:
            db_config = json.load(f)
        user_data = pymysql.connect(host=db_config.get('host'),
                                    user=db_config.get('user'),
                                    password=db_config.get('password'),
                                    port=int(db_config.get('port')),
                                    database=db_config.get('database_name').get('result')
                                    )
    except:
        print("db configuration invalid ")
        sys.exit(0)

    return user_data


def get_user_config(configs):
    sql_connection = load_user_config_sql()
    sql = f'''select section_id, forecast_days from forecast_section_congestion_warning_config where config_id = '{configs}' '''

    cursor = sql_connection.cursor()
    # execute执行操作
    cursor.execute(sql)
    result = cursor.fetchall()
    sql_connection.close()

    # section_id, forecast_days
    # example
    # (('S15-1,S15-2,G2-1,G2-2', 1),)
    # 只使用第一条结果
    return result[0]


def final_reshape(up, down):
    """
    将列表降维
    """
    return np.array(up).flatten().tolist(), np.array(down).flatten().tolist()


def fourier_fit(x, base, *a):
    y = base
    i_max = 1
    # 日变化
    for i in range(1, 3):
        y = y + a[i - 1] * np.cos(2 * np.pi * i * x / 96) + a[-i] * np.sin(2 * np.pi * i * x / 96)
        i_max = i

    # 周变化
    for j in range(1, 4):
        y = y + a[j + i_max - 1] * np.cos(2 * np.pi * j * x / 96 * 7) + a[-j - i_max] * np.sin(
            2 * np.pi * j * x / 96 * 7)

    return y


def fourier_pre_process(data_up, data_down):
    """
    数据格式预处理
    """
    # 读取本地数据以供测试
    # data_up = pd.read_csv('example_data/up.csv')
    # data_down = pd.read_csv('example_data/down.csv')

    x_group = np.arange(96 * 7)
    flow_up = data_up['flow']
    flow_down = data_down['flow']
    speed_up = data_up['speed']
    speed_down = data_down['speed']

    return x_group, flow_up, flow_down, speed_up, speed_down


def make_prediction_fourier(name, input_data_up, input_data_down):
    """
    参考prophet中的原理, 由于14天的交通数据中, 流量自然增长带来的影响很小，则除去prophet中关于自然增长的拟合, 而仅保留傅里叶级数拟合的部分, 以减少计算时间
    """
    x_group, input_up_flow, input_down_flow, input_up_speed, input_down_speed = fourier_pre_process(input_data_up,
                                                                                                    input_data_down)

    # 估计参数
    popt_flow_up = op.curve_fit(fourier_fit, x_group, input_up_flow, [1.0] * 9)[0]
    popt_flow_down = op.curve_fit(fourier_fit, x_group, input_down_flow, [1.0] * 9)[0]
    popt_speed_up = op.curve_fit(fourier_fit, x_group, input_up_speed, [1.0] * 9)[0]
    popt_speed_down = op.curve_fit(fourier_fit, x_group, input_down_speed, [1.0] * 9)[0]

    # 计算未来值
    x_predict = np.arange(96 * 7, 96 * 21)

    # numpy.ndarray 96*14
    flow_result_up = fourier_fit(x_predict, *popt_flow_up)
    flow_result_down = fourier_fit(x_predict, *popt_flow_down)
    speed_result_up = fourier_fit(x_predict, *popt_speed_up)
    speed_result_down = fourier_fit(x_predict, *popt_speed_down)

    # 整合预测数据, 及计算拥堵指数
    # : return result_up, result_down;
    # list result_xx columns are
    # traffic_flow_total, avg_speed_car, traffic_index
    # shape(14, 24, 3)
    # 14天的24小时的 流量 速度 拥堵指数
    return fourier_fin_process(flow_result_up, flow_result_down, speed_result_up, speed_result_down, name)


def fourier_fin_process(flow_result_up, flow_result_down, speed_result_up, speed_result_down, name):
    """
    整合预测数据, 及计算拥堵指数
    : param xx_xx is flow/speed; list
    len (96*14)
    : param name 用于确定车道数量以计算拥堵指数

    : return result_up, result_down; list
    result_xx columns are
    traffic_flow_total, avg_speed_car, traffic_index
    shape (14, 24, 3)
    14天的24小时的 流量 速度 拥堵指数
    """
    result_up = []
    result_down = []

    # 14天中的96分钟
    for j in range(14):
        day_temp_up = []
        day_temp_down = []
        for i in range(24):
            flow_up = (flow_result_up[4 * i + 96 * j] + flow_result_up[4 * i + 1 + 96 * j] + flow_result_up[
                4 * i + 2 + 96 * j] + flow_result_up[4 * i + 3 + 96 * j])
            speed_up = (speed_result_up[4 * i + 96 * j] + speed_result_up[4 * i + 1 + 96 * j] + speed_result_up[
                4 * i + 2 + 96 * j] + speed_result_up[4 * i + 3 + 96 * j]) / 4
            flow_down = (flow_result_down[4 * i + 96 * j] + flow_result_down[4 * i + 1 + 96 * j] + flow_result_down[
                4 * i + 2 + 96 * j] + flow_result_down[4 * i + 3 + 96 * j])
            speed_down = (speed_result_down[4 * i + 96 * j] + speed_result_down[4 * i + 1 + 96 * j] + speed_result_down[
                4 * i + 2 + 96 * j] + speed_result_down[4 * i + 3 + 96 * j]) / 4

            day_temp_up.append((flow_up, speed_up, traffic_index(flow_up, speed_up, 'hour', name)))
            day_temp_down.append((flow_up, speed_up, traffic_index(flow_down, speed_down, 'hour', name)))

        result_up.append(day_temp_up)
        result_down.append(day_temp_down)

    return result_up, result_down

