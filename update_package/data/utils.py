import pandas as pd
import numpy as np
import datetime
from road_names import names


def time_index_hour(index):
    # 合并时间字符串，用于排序
    # 2021-10-25 12
    date, hour = index.split(' ')
    year, month, day = date.split('-')
    num = int(year + month + day + hour)
    return num


def time_index_minutes(index):
    # 合并时间字符串，用于排序
    # 2021-01-25 18:45:00
    date, hours_minutes = index.split(' ')
    year, month, day = date.split('-')
    hours, minutes, second = hours_minutes.split(':')
    num = int(year + month + day + hours + minutes + second)
    return num


def time_hour(now, count):
    # 计算之前，之后时间对应的时间字符串
    # 2021-10-18 15
    return (datetime.datetime.strptime(now, '%Y-%m-%d %H') + datetime.timedelta(hours=+count)).strftime('%Y-%m-%d %H')


def time_minutes(now, count):
    # 计算之前，之后时间对应的时间字符串
    # 2021-01-25 18:45:00

    # 针对这种特殊情况 2021-01-01 24:00:00, 2021-01-01 23:45:00
    date, hours_minutes = now.split(' ')
    if hours_minutes == '24:00:00':
        now = (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=+1)).strftime(
            '%Y-%m-%d') + ' 00:00:00'

    next = (datetime.datetime.strptime(now, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=+count)).strftime(
        '%Y-%m-%d %H:%M:%S')
    n_date, n_hours_minutes = next.split(' ')
    if n_hours_minutes == '00:00:00':
        next = date + ' 24:00:00'

    return next


def time_sin(index):
    # 2021-01-25 18:45:00
    date, hours_minutes = str(index).split(' ')
    hours, minutes, second = hours_minutes.split(':')
    index = float(hours + minutes) * np.pi / 2400
    # print(index)
    return np.sin(index)


def time_cos(index):
    # 2021-01-25 18:45:00
    date, hours_minutes = str(index).split(' ')
    hours, minutes, second = hours_minutes.split(':')
    index = float(hours + minutes) * np.pi / 2400
    # print(index)
    return np.cos(index)


def time_sin_hour(index):
    # 2021-01-25 18
    date, hours = index.split(' ')
    index = float(hours) * np.pi / 24
    # print(index)
    return np.sin(index)


def time_cos_hour(index):
    # 2021-01-25 18
    date, hours = index.split(' ')
    index = float(hours) * np.pi / 24
    # print(index)
    return np.cos(index)


# def hour():
#     #   0,     1,          2,          3,         4,                     5,         6,          7
#     # index,section_id,direction,point_time,    traffic_flow_total,avg_speed_car,time_calc,     if_next_time
#     # 1,        G1-1,       上行, 2021-10-25 09,      324,                100,    2021102509,     0/1
#     # if_next_time值为1表示改行与下一行之间没有空缺值
#     data = pd.read_csv('hour_data.csv')
#
#     for name in names:
#         print(name)
#         select = data.loc[data['section_id'] == name]
#         select['time_calc'] = 1
#         select['if_next_time'] = 1
#         select['hour_sin'] = 1
#         select['hour_cos'] = 1
#         for i in range(select.shape[0]):
#             select.iloc[i, 6] = time_index_hour(select.iloc[i, 3])
#             select.iloc[i, 8] = time_sin_hour(select.iloc[i, 3])
#             select.iloc[i, 9] = time_cos_hour(select.iloc[i, 3])
#
#         # print(select.head())
#         select_up = select.loc[select['direction'] == '上行']
#         select_down = select.loc[select['direction'] == '下行']
#
#         select_up.sort_values(by=['time_calc'], ascending=True, inplace=True)
#         select_down.sort_values(by=['time_calc'], ascending=True, inplace=True)
#
#         for i in range(select_up.shape[0] - 1):
#             if select_up.iloc[i + 1, 3] != time_hour(select_up.iloc[i, 3], 1):
#                 select_up.iloc[i, 7] = 0
#
#         for i in range(select_down.shape[0] - 1):
#             if select_down.iloc[i + 1, 3] != time_hour(select_down.iloc[i, 3], 1):
#                 select_down.iloc[i, 7] = 0
#
#         # print(select.head())
#         # select['time_calc'] = time_index_hour()
#
#         select_up.to_csv('hour_output/' + name + '_up.csv', index=False)
#         select_down.to_csv('hour_output/' + name + '_down.csv', index=False)


# def minute():
#     #   0,     1,          2,          3,                       4,               5,         6,              7           8                   9
#     # index,section_id,direction,point_time,            traffic_flow_total,avg_speed_car,time_calc,     if_next_time, hour_minutes_sin, hour_minutes_cos
#     # 1,        G1-1,       上行, 2021-01-25 18:45:00,      324,                100,      20210125184500,     0/1
#     # if_next_time值为1表示改行与下一行之间没有空缺值
#     data = pd.read_csv('15min_data.csv')
#
#     for name in names:
#         print(name)
#         select = data.loc[data['section_id'] == name]
#         select['time_calc'] = 1
#         select['if_next_time'] = 1
#         select['hour_minutes_sin'] = 1
#         select['hour_minutes_cos'] = 1
#         for i in range(select.shape[0]):
#             select.iloc[i, 6] = time_index_minutes(select.iloc[i, 3])
#             select.iloc[i, 8] = time_sin(select.iloc[i, 3])
#             select.iloc[i, 9] = time_cos(select.iloc[i, 3])
#
#         # print(select.head())
#         select_up = select.loc[select['direction'] == '上行']
#         select_down = select.loc[select['direction'] == '下行']
#
#         select_up.sort_values(by=['time_calc'], ascending=True, inplace=True)
#         select_down.sort_values(by=['time_calc'], ascending=True, inplace=True)
#
#         for i in range(select_up.shape[0] - 1):
#             if select_up.iloc[i + 1, 3] != time_minutes(select_up.iloc[i, 3], 15):
#                 select_up.iloc[i, 7] = 0
#
#         for i in range(select_down.shape[0] - 1):
#             if select_down.iloc[i + 1, 3] != time_minutes(select_down.iloc[i, 3], 15):
#                 select_down.iloc[i, 7] = 0
#
#         # print(select.head())
#         # select['time_calc'] = time_index_hour()
#
#         select_up.to_csv('15min_output/' + name + '_up.csv', index=False)
#         select_down.to_csv('15min_output/' + name + '_down.csv', index=False)


if __name__ == '__main__':
    pass
