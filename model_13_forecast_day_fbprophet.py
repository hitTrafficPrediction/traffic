import sys
import pandas as pd
from utils.api import read_local_data, days_interval, day_data_process, make_prediction_prophet, next_day
from utils.time_processor import get_day_start_time, get_current_proximity_time_str
from utils.db_operator import writehour_data, get15min_data, get15min_data_week


def main(state, custom_day):
    """
    使用过去一个星期的数据拟合未来两个星期的数据
    利用prophet的日周期变化与周周期变化

    :参数 state 与接口文档示例般的字典格式，定义运行参数。
    示例：
    未来一周：
    {
	    "trace_id": "0ca175b9c0f726a831d895e269332461",
	    "expressway_number": "S15",
	    "section_id": "S15-1,S15-2",
	    "custom_days": {}
    }
    自定义日期范围：
    {
	    "trace_id": "0ca175b9c0f726a831d895e269332461",
	    "expressway_number": "S15",
	    "section_id": "S15-1,S15-2",
	    "custom_days": {
		    "start_date": "2021-04-20",
		    "end_date": "2021-04-21"
	        }
    }
    :参数 custom_day str格式
    可传入自定义时间以供测试，留空使用当前的时间
    示例：
    2021-01-25
    """

    # if state['custom_days'] == {}:
    #     prediction_days = 7
    #     custom_time = get_current_proximity_time_str()
    #     custom_time = get_day_start_time(custom_time).strftime("%Y-%m-%d %H")
    # else:
    #     prediction_days = days_interval(state['custom_days']['start_date'], state['custom_days']['end_date'])
    #     custom_time = state['custom_days']['start_date'] + ' 00'

    prediction_days = 14

    if custom_day:
        custom_time = custom_day + ' 00'
    else:
        custom_time = get_current_proximity_time_str()
        custom_time = get_day_start_time(custom_time).strftime("%Y-%m-%d %H")

    custom_time_read = custom_time + ':00:00'

    sections = state['section_id'].split(',')
    for section_id in sections:
        # 进行预测

        section_id = section_id.replace(' ', '')
        # : input_data_xx columns are
        # traffic_flow_total, avg_speed_car, point_time
        # len (96) list of tuple
        # input_data_up, input_data_down = read_local_data(section_id, '15minutes')
        input_data_up, input_data_down = get15min_data_week(section_id, custom_time_read)

        # : result_xx columns are
        # traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
        # example (result_xx[0])
        # (69.0, 100.0, 4) -> traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
        # shape (24) list of tuple
        result_up, result_down = make_prediction_prophet(section_id, input_data_up, input_data_down)
        for i in range(prediction_days):
            writehour_data(state['trace_id'], state, custom_time + ':00:00', result_up[i], result_down[i])
            custom_time = next_day(custom_time, 1)


if __name__ == '__main__':
    state = {
        "trace_id": "64564",
        "expressway_number": "S15",
        "section_id": "S15-1",
        "custom_days": {}
    }
    main(state, '2021-01-25')
