import pandas as pd
from road_names import names
from utils import time_sin, time_cos


def simplify(input_file_name, save_file_name='15min_data.csv'):
    """
    精简原始csv文件中的列,仅保留训练需要的数据列,可根据实际需要修改

    参考:
        -原始列:
        Index(['index', 'expressway_number', 'road_id', 'section_id', 'direction',
       'section_name', 'time_day', 'time_point', 'point_time_stop',
       'traffic_flow_car', 'traffic_flow_truck', 'traffic_flow_total',
       'avg_speed_car', 'avg_speed_truck', 'avg_speed_total', 'calc_time',
       'point_time_start'],
        dtype='object')

        -转换后的列:
        ['index', 'section_id', 'direction', 'point_time_stop', 'traffic_flow_total', 'avg_speed_car']
    """
    # data = pd.read_csv('kpi_section_condition_hour.csv')
    data = pd.read_csv(input_file_name)
    kpi_minutes = data[['index', 'section_id', 'direction', 'point_time_stop', 'traffic_flow_total', 'avg_speed_car']]
    # print(kpi_minutes.head())

    kpi_minutes.to_csv(save_file_name, index=False)


def to_minute_data(data_file_name='15min_data.csv', save_dir='output_data/'):
    """
    将原始数据文件按照路段分离,并添加训练需要的值,并按照时间排序

    -文件名:
        路段名 + '_up.csv'
        路段名 + '_down.csv'
        分别代表上下行

    -转换后的列:
    Index(['index', 'section_id', 'direction', 'point_time_stop',
       'traffic_flow_total', 'avg_speed_car', 'hour_minutes_sin',
       'hour_minutes_cos'],
        dtype='object')
    """
    data = pd.read_csv(data_file_name)

    for name in names:
        print(name)

        # 创建对应路段数据表
        select = data[data['section_id'] == name]

        # 创建新列,并排序
        select.insert(select.shape[1], 'hour_minutes_sin', 1)
        select.insert(select.shape[1], 'hour_minutes_cos', 1)
        select['point_time_stop'] = pd.to_datetime(select['point_time_stop'], format='%Y-%m-%d %H:%M:%S')
        select = select.sort_values(by=['point_time_stop'], ascending=True)
        select = select.reset_index(drop=True)

        for i in range(select.shape[0]):

            select.loc[i, 'hour_minutes_sin'] = time_sin(select.loc[i, 'point_time_stop'])
            select.loc[i, 'hour_minutes_cos'] = time_cos(select.loc[i, 'point_time_stop'])

        # print(select.head())
        select_up = select[select['direction'] == '上行']
        select_down = select[select['direction'] == '下行']

        select_up.sort_values(by=['point_time_stop'], ascending=True)
        select_down.sort_values(by=['point_time_stop'], ascending=True)

        # 保存文件
        select_up.to_csv(save_dir + name + '_up.csv', index=False)
        select_down.to_csv(save_dir + name + '_down.csv', index=False)


if __name__ == '__main__':
    # df = pd.read_csv('kpi_section_condition_15minute.csv')
    # print(df.loc[0])
    # simplify('raw_data/kpi_section_condition_15minute.csv')
    # to_minute_data()
    pass
