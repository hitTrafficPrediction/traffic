from utils.api import make_prediction, read_local_data, days_interval, day_data_process
from utils.time_processor import get_day_start_time, get_current_proximity_time_str
from utils.db_operator import writeday_data, get15min_data




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

        # : result_xx columns are
        # traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
        # example (result_xx[0])
        # (69.0, 100.0, 4) -> traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
        # shape (24) list of tuple
        result_up, result_down = make_prediction(section_id, device, 'hour', input_data_up, input_data_down)

        # : processed_xx columns are
        # peak_value, normal_value, valley_value tuples for flow, speed, and index
        # example (processed_xx[0])
        # ( (77.38573605907594, 90.88928731770665, 4),      peak -> flow,speed,index(拥堵指数）
        #   (74.39937376149416, 90.88928731770665, 4),      normal -> flow,speed,index(拥堵指数）
        #   (70.71186553317577, 90.88928731770665, 4)   )   valley -> flow,speed,index(拥堵指数）
        # len(processed_xx) = prediction_days
        processed_up, processed_down = day_data_process(result_up, result_down, prediction_days, section_id)
        # print(processed_up)
        writeday_data(state['trace_id'], state, custom_time.replace(' 00', ''), processed_up, processed_down)


if __name__ == '__main__':
    state = {
        "trace_id": "057",
        "expressway_number": "S15",
        "section_id": "S15-1",
        "custom_days": {}
    }
    main(state, '2021-01-25')
