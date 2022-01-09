import numpy
import datetime
from utils.time_processor import get_time_backward_15min_sequence, get_time_backward_hour_sequence, \
    get_time_forward_15min_sequence, get_day_from_time_str, get_time_point_from_time_str, get_day_start_time_str, \
    get_time_forward_hour_sequence, get_time_forward_day_sequence, get_time_forward_30min_sequence

db_config = {}
db_data = None
db_result = None

import pymysql
import json


def init_db():
    global db_config
    global db_data
    global db_result
    try:
        with open("config/db_config.json") as f:
            db_config = json.load(f)
        db_data = pymysql.connect(host=db_config.get('host'),
                                  user=db_config.get('user'),
                                  password=db_config.get('password'),
                                  port=int(db_config.get('port')),
                                  database=db_config.get('database_name').get('raw_data')
                                  )
        db_result = pymysql.connect(host=db_config.get('host'),
                                    user=db_config.get('user'),
                                    password=db_config.get('password'),
                                    port=int(db_config.get('port')),
                                    database=db_config.get('database_name').get('result')
                                    )
    except:
        print("db configuration invalid ")


def next_day(now, count):
    # 计算之前，之后时间对应的时间字符串
    # 2021-01-25
    next_time = (datetime.datetime.strptime(now, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=+count)).strftime(
        '%Y-%m-%d %H:%M:%S')

    return next_time


def get15min_data_week(section_id, custom_time):
    result_up = []
    result_down = []

    custom_time = next_day(custom_time, -6)
    for i in range(7):
        data_up, data_down = get15min_data(section_id, custom_time)
        for item_up in data_up:
            result_up.append(item_up)
        for item_down in data_down:
            result_down.append(item_down)

        custom_time = next_day(custom_time, 1)

    return result_up, result_down


def get15min_data(road_section, custom_time):
    init_db()
    cursor_data = db_data.cursor()
    data_table_15min = "section_condition_15minute"

    # 使得输入可以为24:00:00
    # custom_data = custom_time.split(' ')[0]
    # custom_time = custom_time.split(' ')[1].replace('24', '00')
    # custom_time = custom_data + ' ' + custom_time
    time_seq = get_time_backward_15min_sequence(custom_time, 96)
    # end of edit

    # 将查询数据库的对应时间改为24:00:00
    # for idx, item in enumerate(time_seq):
    #     custom_time = item.split(' ')[1]
    #     if custom_time == '00:00:00':
    #         custom_time = '24:00:00'
    #     time_seq[idx] = item.split(' ')[0] + ' ' + custom_time
    time_seq = '\',\''.join(time_seq)
    # end of edit

    get15min_SQL_down = f'''
    SELECT traffic_flow_total,avg_speed_car, point_time_start
    FROM {data_table_15min}
    WHERE section_id = '{road_section}' AND direction = '下行' AND point_time_start in ('{time_seq}')
    ORDER BY point_time_start
    '''
    get15min_SQL_up = f'''
    SELECT traffic_flow_total,avg_speed_car, point_time_start
    FROM {data_table_15min}
    WHERE section_id = '{road_section}' AND direction = '上行' AND point_time_start in ('{time_seq}')
    ORDER BY point_time_start
    '''
    data_up = []
    data_down = []
    try:
        # 执行sql
        cursor_data.execute(get15min_SQL_down)
        data_down = cursor_data.fetchall()
        cursor_data.execute(get15min_SQL_up)
        data_up = cursor_data.fetchall()
    except Exception as e:
        print(e)
        print('插入失败')
    finally:
        db_data.close()
    assert (len(data_up) == 96 and len(data_down)), '数据缺失'
    return [data_up, data_down]
    # return data


def write15min_data(trace_id, state, prediction_start_time, result_up, result_down):
    init_db()
    cursor_result = db_result.cursor()
    result_table_15min = "forecast_section_flow_speed_index_15minutes"
    time_seq = get_time_forward_15min_sequence(prediction_start_time, 96)
    result_up_processed = []
    result_down_processed = []
    for idx, item in enumerate(result_up):
        traffic_flow, avg_speed, traffic_index = item
        item_time = time_seq[idx]
        time_day = get_day_from_time_str(item_time)
        time_point = get_time_point_from_time_str(item_time, get_day_start_time_str(item_time), 15)
        result_up_processed.append(
            f"('{trace_id}','{state['expressway_number']}','{state['section_id']}',0,'{time_day}',{time_point},'{item_time}',{traffic_flow},{avg_speed},{traffic_index})")
    for idx, item in enumerate(result_down):
        traffic_flow, avg_speed, traffic_index = item
        item_time = time_seq[idx]
        time_day = get_day_from_time_str(item_time)
        time_point = get_time_point_from_time_str(item_time, get_day_start_time_str(item_time), 15)
        result_down_processed.append(
            f"('{trace_id}','{state['expressway_number']}','{state['section_id']}',1,'{time_day}',{time_point},'{item_time}','{traffic_flow}',{avg_speed},{traffic_index})")
    final_result = result_down_processed + result_up_processed
    final_result = ',\n'.join(final_result)
    write15min_SQL = f'''
        INSERT INTO {result_table_15min}(trace_id,expressway_number,section_id,direction,time_day,time_point,point_time,traffic_flow,avg_speed,traffic_index)
        VALUES {final_result}
    '''

    try:
        # 执行sql
        cursor_result.execute(write15min_SQL)
        # 提交事务
        db_result.commit()
        print('插入成功')
    except Exception as e:
        print(e)
        db_result.rollback()
        print('插入失败')
    finally:
        db_result.close()


def writehour_data(trace_id, state, prediction_start_time, result_up, result_down):
    init_db()
    cursor_result = db_result.cursor()
    result_table_hour = "forecast_section_flow_speed_index_hour"
    time_seq = get_time_forward_hour_sequence(prediction_start_time, 24)
    result_up_processed = []
    result_down_processed = []
    for idx, item in enumerate(result_up):
        traffic_flow, avg_speed, traffic_index = item
        item_time = time_seq[idx]
        time_day = get_day_from_time_str(item_time)
        time_point = get_time_point_from_time_str(item_time, get_day_start_time_str(item_time), 60)
        result_up_processed.append(
            f"('{trace_id}','{state['expressway_number']}','{state['section_id']}',0,'{time_day}',{time_point},'{time_day} {time_point}',{traffic_flow},{avg_speed},{traffic_index})")
    for idx, item in enumerate(result_down):
        traffic_flow, avg_speed, traffic_index = item
        item_time = time_seq[idx]
        time_day = get_day_from_time_str(item_time)
        time_point = get_time_point_from_time_str(item_time, get_day_start_time_str(item_time), 60)
        result_down_processed.append(
            f"('{trace_id}','{state['expressway_number']}','{state['section_id']}',1,'{time_day}',{time_point},'{time_day} {time_point}',{traffic_flow},{avg_speed},{traffic_index})")
    final_result = result_down_processed + result_up_processed
    final_result = ',\n'.join(final_result)
    writehour_SQL = f'''
            INSERT INTO {result_table_hour}(trace_id,expressway_number,section_id,direction,time_day,time_point,point_time,traffic_flow,avg_speed,traffic_index)
            VALUES {final_result}
        '''
    cursor_result.execute(writehour_SQL)
    try:
        # 执行sql
        cursor_result.execute(writehour_SQL)
        # 提交事务
        db_result.commit()
        print('插入成功')
    except Exception as e:
        print(e)
        db_result.rollback()
        print('插入失败')
    finally:
        db_result.close()


def writeday_data(trace_id, state, start_day, result_up, result_down):
    init_db()
    cursor_result = db_result.cursor()
    result_table_day = "forecast_section_flow_speed_index_day"
    time_seq = get_time_forward_day_sequence(start_day + ' ' + '00:00:00', len(result_up))
    result_up_processed = []
    result_down_processed = []
    for idx, item in enumerate(result_up):
        item_time = time_seq[idx]
        time_day = get_day_from_time_str(item_time).strftime('%Y-%m-%d')
        peak = item[0]
        normal = item[1]
        valley = item[2]
        item_tuple = (
            trace_id, state['expressway_number'], state['section_id'], 0, time_day, peak[0], normal[0], valley[0],
            peak[0] + normal[0] + valley[0], peak[1], normal[1], valley[1], (peak[1] + normal[1] + valley[1]) / 3,
            peak[2],
            normal[2], valley[2], (peak[2] + valley[2] + normal[2]) / 3)
        result_up_processed.append(item_tuple)
    for idx, item in enumerate(result_down):
        item_time = time_seq[idx]
        time_day = get_day_from_time_str(item_time).strftime('%Y-%m-%d')
        peak = item[0]
        normal = item[1]
        valley = item[2]
        item_tuple = (
            trace_id, state['expressway_number'], state['section_id'], 1, time_day, peak[0], normal[0], valley[0],
            peak[0] + normal[0] + valley[0], peak[1], normal[1], valley[1], (peak[1] + normal[1] + valley[1]) / 3,
            peak[2],
            normal[2], valley[2], (peak[2] + valley[2] + normal[2]) / 3)
        result_down_processed.append(item_tuple)
    writeday_SQL = f'''
            INSERT INTO {result_table_day} (trace_id,expressway_number,section_id,direction,time_day,peak_flow,normal_flow,valley_flow,total_flow,peak_speed,normal_speed,valley_speed,total_speed,peak_index,normal_index,valley_index,total_index)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        '''

    try:
        # 执行sql
        cursor_result.executemany(writeday_SQL, result_up_processed)
        cursor_result.executemany(writeday_SQL, result_down_processed)
        # 提交事务
        db_result.commit()
        print('插入成功')
    except Exception as e:
        print(e)
        db_result.rollback()
        print('插入失败')
    finally:
        db_result.close()


def write_festival_data(trace_id, state, prediction_start_time, result_up, result_down, type):
    init_db()
    cursor_result = db_result.cursor()
    result_table_day = "forecast_holiday_section_condition"
    time_delta = 0
    time_count = 0
    if type == 'hour':
        time_delta = 60
        time_count = 24
        time_seq = get_time_forward_hour_sequence(prediction_start_time, time_count * len(result_up))
    elif type == '30min':
        time_delta = 30
        time_count = 48
        time_seq = get_time_forward_30min_sequence(prediction_start_time, time_count * len(result_up))
    result_up_processed = []
    result_down_processed = []
    for day_id, day_item in enumerate(result_up):
        for time_id, time_item in enumerate(day_item):
            item_time = time_seq[day_id * time_count + time_id]
            granularity = type
            expressway_number = state['expressway_number']
            section_id = state['section_id']
            direction = 0
            time_day = get_day_from_time_str(item_time).strftime('%Y-%m-%d')
            time_point = time_id
            point_time = item_time
            traffic_flow = time_item[0]
            avg_speed = time_item[1]
            traffic_index = time_item[2]
            item_tuple = (
                granularity, trace_id, expressway_number, section_id, direction, time_day, time_point, point_time,
                traffic_flow, avg_speed, traffic_index)
            result_up_processed.append(item_tuple)
    for day_id, day_item in enumerate(result_down):
        for time_id, time_item in enumerate(day_item):
            item_time = time_seq[day_id * time_count + time_id]
            granularity = type
            expressway_number = state['expressway_number']
            section_id = state['section_id']
            direction = 1
            time_day = get_day_from_time_str(item_time).strftime('%Y-%m-%d')
            time_point = time_id
            point_time = item_time
            traffic_flow = time_item[0]
            avg_speed = time_item[1]
            traffic_index = time_item[2]
            item_tuple = (
                granularity, trace_id, expressway_number, section_id, direction, time_day, time_point, point_time,
                traffic_flow, avg_speed, traffic_index)
            result_down_processed.append(item_tuple)
    writefestival_SQL = f'''
               INSERT INTO {result_table_day} (granularity,trace_id,expressway_number,section_id,direction,time_day,time_point,point_time,traffic_flow,avg_speed,traffic_index)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           '''
    try:
        # 执行sql
        cursor_result.executemany(writefestival_SQL, result_up_processed)
        cursor_result.executemany(writefestival_SQL, result_down_processed)
        # 提交事务
        db_result.commit()
        print('插入成功')
    except Exception as e:
        print(e)
        db_result.rollback()
        print('插入失败')
    finally:
        db_result.close()


def read_congestion_config(config_id):
    init_db()
    cursor_result = db_result.cursor()
    data_table_congestion_config = "forecast_section_congestion_warning_config"
    get_congestion_SQL = f'''
        SELECT *
        FROM {data_table_congestion_config}
        WHERE config_id = '{config_id}'
        '''
    cursor_result.execute(get_congestion_SQL)
    data = cursor_result.fetchall()[0]
    return data


def write_congestion_data(trace_id, state, prediction_start_time, result_up, result_down, threshold):
    init_db()
    cursor_result = db_result.cursor()
    # 传入参数示例，多个路段请多次调用
    # state = {
    #     "trace_id": "0ca175b9c0f726a831d895e269332461",
    #     "expressway_number": "S15",
    #     "section_id": "G1-1",
    #     "prediction_days":2
    # }
    result_table_congestion_brief = "forecast_section_congestion_warning_brief"
    result_table_congestion_detail = "forecast_section_congestion_warning_detail"
    prediction_days = state['prediction_days']
    detail_result_up_processed = []
    detail_result_down_processed = []
    time_seq = get_time_forward_30min_sequence(prediction_start_time, 48 * prediction_days)
    warning_period_up = []
    warning_period_down = []
    time_day = ''
    for idx, item in enumerate(result_up):
        traffic_index = item
        if traffic_index >= threshold:
            time_day = get_day_from_time_str(time_seq[idx])
            time_point = get_time_point_from_time_str(time_seq[idx], get_day_start_time_str(time_seq[idx]), 30)
            detail_item = (
                trace_id, state['expressway_number'], state['section_id'], 0, time_day, time_point, time_seq[idx],
                traffic_index)
            detail_result_up_processed.append(detail_item)
            warning_period_up.append(time_seq[idx])
    brief_item_up = (
        trace_id, state['expressway_number'], state['section_id'], 0, get_day_from_time_str(prediction_start_time),
        "[" + str(warning_period_up) + "]")

    for idx, item in enumerate(result_down):
        traffic_index = item
        if traffic_index >= threshold:
            time_day = get_day_from_time_str(time_seq[idx])
            time_point = get_time_point_from_time_str(time_seq[idx], get_day_start_time_str(time_seq[idx]), 30)
            detail_item = (
                trace_id, state['expressway_number'], state['section_id'], 1, time_day, time_point, time_seq[idx],
                traffic_index)
            detail_result_down_processed.append(detail_item)
            warning_period_down.append(time_seq[idx])
    brief_item_down = (
        trace_id, state['expressway_number'], state['section_id'], 1, get_day_from_time_str(prediction_start_time),
        "[" + str(warning_period_down) + "]")

    write_congestion_brief = f'''
                   INSERT INTO {result_table_congestion_brief} (trace_id,expressway_number,section_id,direction,time_day,warning_period)
                   VALUES (%s,%s,%s,%s,%s,%s)
               '''
    write_congestion_detail = f'''
                   INSERT INTO {result_table_congestion_detail} (trace_id,expressway_number,section_id,direction,time_day,time_point,point_time,traffic_index)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
               '''
    try:
        # 执行sql
        cursor_result.executemany(write_congestion_detail, detail_result_up_processed)
        cursor_result.executemany(write_congestion_detail, detail_result_down_processed)
        cursor_result.execute(write_congestion_brief, brief_item_down)
        cursor_result.execute(write_congestion_brief, brief_item_up)
        # 提交事务
        db_result.commit()
        print('插入成功')
    except Exception as e:
        print(e)
        db_result.rollback()
        print('插入失败')
    finally:
        db_result.close()


if __name__ == "__main__":
    # init_db()
    result_up = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    result_down = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    state = {
        "trace_id": "0ca175b9c0f726a831d895e269332461",
        "expressway_number": "S15",
        "section_id": "G1-1",
        "prediction_days": 2
    }
    trace_id = '0ca175b9c0f726a831d895e269332461'
    # write_festival_data(trace_id, state, '2022-03-25 00:00:00', result_up, result_down,'hour')
    get15min_data('G1-1', '2021-01-25 18:30:00')
