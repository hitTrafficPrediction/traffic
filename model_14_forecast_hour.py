from utils.db_operator import write_festival_data, get15min_data
from utils.api import make_prediction, read_local_data, days_interval, holiday_data_process_hour
from utils.time_processor import get_day_start_time, get_current_proximity_time_str
import datetime as dt


def main(state, custom_day):
    """
    :参数 state 与接口文档示例般的字典格式，定义运行参数。
    示例：
    {
	    "trace_id": "0ca175b9c0f726a831d895e269332461",
    	"expressway_number": "S15",
    	"section_id": "S15-1,S15-2",
    	"holiday": {
	    	"name": "guoqing",
	    	"start_date": "2021-10-01",
		"end_date": "2021-10-07"
	    }
    }
    :参数 custom_day str格式
    可传入自定义时间以供测试，留空使用当前的时间
    示例：
    2021-01-25
    """

    prediction_days = days_interval(state['holiday']['start_date'], state['holiday']['end_date'])
    # 获取当前时间以供预测
    custom_time = get_current_proximity_time_str()
    custom_time = get_day_start_time(custom_time).strftime("%Y-%m-%d %H")

    if custom_day:
        custom_time = custom_day + ' 00'

    # 可根据实际情况指定设备为 cuda 而使用GPU
    device = 'cpu'

    sections = state['section_id'].split(',')
    for section_id in sections:
        # 进行预测

        section_id = section_id.replace(' ', '')
        # : input_data_xx columns are
        # traffic_flow_total, avg_speed_car, point_time
        # len (96) list of tuple
        # input_data_up, input_data_down = read_local_data(section_id, '15minutes')
        input_data_up, input_data_down = get15min_data(section_id, custom_time)

        # : result_xx columns are
        # traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
        # example (result_xx[0])
        # (69.0, 100.0, 4) -> traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
        # shape (24) list of tuple
        result_up, result_down = make_prediction(section_id, device, 'hour', input_data_up, input_data_down)

        # : processed_xx columns are
        # 24 粒度 of (traffic_flow_total, avg_speed_car, traffic_index(拥堵指数) ) a day
        # example (result_xx[0], which is the first day's data)
        # [79.41736302394557, 118.53337764767996, 4.0], 00:00's data for traffic_flow_total, avg_speed_car, traffic_index(拥堵指数)
        # [90.08536701223676, 118.53337764767996, 4.0], 01:00's data
        # ...
        # [99.56803722405115, 118.53337764767996, 4.0], 23:00's data
        # [96.01203589462077, 118.53337764767996, 4.0], 24:00's data
        # len(processed_xx) ==  prediction_days+6, list of tuple/list
        processed_up, processed_down = holiday_data_process_hour(result_up, result_down, prediction_days, section_id)

        start_time = (dt.datetime.strptime(state["holiday"]["start_date"], '%Y-%m-%d') + dt.timedelta(days=-3)).strftime('%Y-%m-%d')
        write_festival_data(state['trace_id'], state, start_time + ' 00:00:00', processed_up, processed_down, 'hour')


if __name__ == '__main__':
    state = {
        "trace_id": "1124",
        "expressway_number": "G1",
        "section_id": "G1-1",
        "holiday": {
            "name": "guoqing",
            "start_date": "2021-10-01",
            "end_date": "2021-10-07"
        }
    }
    main(state, None)
