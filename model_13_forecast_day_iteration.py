from utils.api import make_prediction_iteration, read_local_data, days_interval, day_data_process, data_Pre_process, next_day, show_img
from utils.time_processor import get_day_start_time, get_current_proximity_time_str
from utils.db_operator import writehour_data, get15min_data


def main(state, custom_day):
    """
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

    if state['custom_days'] == {}:
        prediction_days = 7
        custom_time = get_current_proximity_time_str()
        custom_time = get_day_start_time(custom_time).strftime("%Y-%m-%d %H")
    else:
        prediction_days = days_interval(state['custom_days']['start_date'], state['custom_days']['end_date'])
        custom_time = state['custom_days']['start_date'] + ' 00'

    if custom_day:
        custom_time = custom_day + ' 00'

    custom_time_read = custom_time + ':00:00'
    # 可根据实际情况指定设备为 cuda 而使用GPU#
    device = 'cpu'

    sections = state['section_id'].split(',')
    for section_id in sections:
        # 进行预测

        section_id = section_id.replace(' ', '')
        # : input_data_xx columns are
        # traffic_flow_total, avg_speed_car, point_time
        # len (96) list of tuple
        # input_data_up, input_data_down = read_local_data(section_id, '15minutes')
        input_data_up, input_data_down = get15min_data(section_id, custom_time_read)

        # 数据预处理
        # traffic_flow_total, avg_speed_car, sin(point_time), cos(point_time)
        # shape(1, 1, time_step, 4)
        input_up, input_down = data_Pre_process(input_data_up, input_data_down)

        # 更新 迭代器
        for i in range(prediction_days):
            # : result_xx columns are
            # traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
            # example (result_xx[0])
            # (69.0, 100.0, 4) -> traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
            # shape (24) list of tuple
            result_up, result_down, next_up, next_down = make_prediction_iteration(section_id, device, 'hour', input_up, input_down)

            # show_img(result_up, 'index')
            # show_img(result_up, 'flow')
            # show_img(result_up, 'speed')
            writehour_data(state['trace_id'], state, custom_time + ':00:00', result_up, result_down)

            input_up = next_up
            input_down = next_down
            custom_time = next_day(custom_time, 1)


if __name__ == '__main__':
    state = {
        "trace_id": "057",
        "expressway_number": "S15",
        "section_id": "S15-2",
        "custom_days": {}
    }
    main(state, '2021-10-26')
