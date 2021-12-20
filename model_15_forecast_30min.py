from utils.api import make_prediction, read_local_data, get_user_config, congestion_data_process, next_time_minutes, \
    final_reshape
from utils.time_processor import get_day_start_time, get_current_proximity_time_str
from utils.db_operator import write_congestion_data, get15min_data


def main(state, threshold, custom_day):
    """
    :参数 state 与接口文档示例般的字典格式，定义运行参数。
    示例：
    通过config_id从数据库中获取，用户关心的路段及预测天数
    {
	"trace_id": "0ca175b9c0f726a831d895e269332461",
	"config_id": "hfigfvigibios2784328"，
    }
    """

    # example
    # ('S15-1,S15-2,G2-1,G2-2', 1)
    user_info = get_user_config(state['config_id'])

    prediction_days = user_info[1]
    # 获取当前时间以供预测
    # 再往前1hour, 确保数据可获得性
    custom_time = next_time_minutes(get_current_proximity_time_str(), -60)
    custom_time = get_day_start_time(custom_time).strftime("%Y-%m-%d %H")

    if custom_day:
        custom_time = custom_day + ' 00'

    # 可根据实际情况指定设备为 cuda 而使用GPU
    device = 'cpu'

    sections = user_info[0].split(',')
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
        # shape (96) list of tuple
        result_up, result_down = make_prediction(section_id, device, '15minutes', input_data_up, input_data_down)

        # : processed_xx columns are
        # 48 粒度 of traffic_index(拥堵指数)
        # example (result_xx[0])
        # [1, 1, ... 5, 3]
        processed_up, process_down = congestion_data_process(result_up, result_down, prediction_days, section_id)

        reshape_up, reshape_down = final_reshape(processed_up, process_down)

        expressway_number, _ = section_id.split('-')
        n_state = {
            "trace_id": state['trace_id'],
            "expressway_number": expressway_number,
            "section_id": section_id,
            "prediction_days": prediction_days
        }

        write_congestion_data(state['trace_id'], n_state, custom_time + ':00:00', reshape_up, reshape_down, threshold)


if __name__ == '__main__':
    state = {
        "trace_id": "6671",
        "config_id": "hfigfvigibios2784328",
    }
    # 拥堵指数阈值
    threshold = 1 (大于或等于)
    main(state, threshold, None)

    '''
    user_info = get_user_config('hfigfvigibios2784328')

    print(user_info)
    print(type(user_info))
    print(len(user_info))
    print(user_info[0])
    print(type(user_info[0]))
    print(user_info[0][0])
    '''
