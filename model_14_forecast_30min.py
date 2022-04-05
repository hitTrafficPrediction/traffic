from utils.api import make_prediction_prophet_n, read_local_data, days_interval, holiday_data_process, next_day
from utils.time_processor import get_day_start_time, get_current_proximity_time_str
from utils.db_operator import write_festival_data, get15min_data, writehour_data, get15min_data_week
import datetime as dt


def main(state, custom_day):
    """

    基于当前日期对节假日进行预测，故逻辑上距离节假日约近准确度越高

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
    start_time = next_day(str(state['holiday']['start_date'] + ' 00'), -3)

    if custom_day:
        custom_time = custom_day + ' 00'

    custom_time_read = custom_time + ':00:00'

    # 可根据实际情况指定设备为 cuda 而使用GPU
    device = 'cpu'

    sections = state['section_id'].split(',')
    for section_id in sections:
        # 进行预测

        section_id = section_id.replace(' ', '')
        # : input_data_xx columns are
        # traffic_flow_total, avg_speed_car, point_time
        # len (96*7) list of tuple
        # input_data_up, input_data_down = read_local_data(section_id, '15minutes')
        input_data_up, input_data_down = get15min_data_week(section_id, custom_time_read)

        # : result_xx columns are
        # traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
        # example (result_xx[0])
        # (69.0, 100.0, 4) -> traffic_flow_total, avg_speed_car, traffic_index(拥堵指数)
        # shape (48) list of tuple
        result_up, result_down = make_prediction_prophet_n(section_id, '30minutes', input_data_up, input_data_down, prediction_days)

        write_festival_data(state['trace_id'], state, start_time + ':00:00', result_up, result_down, '30min')


if __name__ == '__main__':
    state = {
        "trace_id": "2022-04-05-0002",
        "expressway_number": "S15",
        "section_id": "S15-1",
        "holiday": {
            "name": "guoqing",
            "start_date": "2021-10-01",
            "end_date": "2021-10-07"
        }
    }
    main(state, '2021-02-09')
