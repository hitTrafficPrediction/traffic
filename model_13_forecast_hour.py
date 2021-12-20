from utils.api import make_prediction, read_local_data, next_time_minutes
from utils.time_processor import get_current_proximity_time_str
from utils.db_operator import writehour_data,get15min_data


def main(state, custom_time):
    """
    :参数 state 与接口文档示例般的字典格式，定义运行参数。
    示例：
    {
    "trace_id": "0ca175b9c0f726a831d895e269332461",
    "expressway_number": "S15",
    "section_id": "S15-1,S15-2",
    "custom_days": {}
    }
    :参数 custom_time str格式
    可传入自定义时间以供测试，留空使用当前的时间
    示例：
    2021-01-25 18
    """
    # 可传入自定义时间以供测试，留空使用当前的时间
    if custom_time is None:
        # 再往前1hour, 确保数据可获得性
        custom_time = next_time_minutes(get_current_proximity_time_str(), -60)
        custom_time, _ = custom_time.split(':', 1)
        # print(custom_time)
    # 可根据实际情况指定设备为 cuda 而使用GPU
    device = 'cpu'

    sections = state['section_id'].split(',')
    for section_id in sections:
        # 进行预测

        section_id = section_id.replace(' ', '')
        # : input_data_xx columns are
        # traffic_flow_total, avg_speed_car, point_time
        # shape (96) list of tuple
        # input_data_up, input_data_down = read_local_data(section_id, '15minutes')
        input_data_up, input_data_down = get15min_data(section_id, custom_time)

        # : result_xx columns are
        # traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
        # example (result_xx[0])
        # (69.0, 100.0, 4) -> traffic_flow_total, avg_speed_car, traffic_index(拥堵指数）
        # shape (24) list of tuple
        result_up, result_down = make_prediction(section_id, device, 'hour', input_data_up, input_data_down)
        # print(result_down)

        writehour_data(state['trace_id'], state, custom_time + ':00:00', result_up, result_down)


if __name__ == '__main__':
    state = {
        "trace_id": '012',
        "expressway_number": "S15",
        "section_id": "S15-1",
        "custom_days": {}
    }
    main(state, '2021-01-25 18')
    # print(next_time_minutes(get_current_proximity_time_str(), -15))

