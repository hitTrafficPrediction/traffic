import datetime as dt


# 获取当前时间字符串%Y-%m-%d %H:%M:%S.%f
def get_current_time_str():
    current_time = dt.datetime.now()
    return dt.datetime.strftime(current_time, "%Y-%m-%d %H:%M:%S.%f")


# 获取当前日期字符串
def get_current_day_str():
    current_day = dt.date.today()
    return current_day.strftime("%Y-%m-%d")


# 获取每天以15min为区间的大约时间
def get_current_proximity_time_str():
    date = dt.datetime.now().date()
    hour = dt.datetime.now().hour
    minute = int(dt.datetime.now().minute / 15) * 15
    time = dt.time(hour, minute, 00)
    proximity_time = dt.datetime.combine(date, time)
    return proximity_time.strftime("%Y-%m-%d %H:%M:%S")


# 获取每天以1hour为区间的大约时间
def get_current_proximity_hour_str():
    date = dt.datetime.now().date()
    hour = dt.datetime.now().hour
    minute = 00
    time = dt.time(hour, minute, 00)
    proximity_time = dt.datetime.combine(date, time)
    # return proximity_time.strftime("%Y-%m-%d %H:%M:%S")
    return proximity_time.strftime("%Y-%m-%d %H")


# 获取每天的零点时间
# 此处还没有改为24:00:00格式
def get_day_start_time(time_str):
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    date = origin_time.date()
    time = dt.time(00, 00, 00)
    day_start_time = dt.datetime.combine(date, time)
    return day_start_time


# 获取每天的零点时间字符串
# 此处还没有改为24:00:00格式
def get_day_start_time_str(time_str):
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    date = origin_time.date()
    time = dt.time(00, 00, 00)
    day_start_time = dt.datetime.combine(date, time)
    return day_start_time.strftime("%Y-%m-%d %H:%M:%S")


# 获取15min前的时间的字符串
def get_time_backward_15min_str(time_str):
    time_gap = dt.timedelta(minutes=15)
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    result = origin_time - time_gap
    result_time = result.time()
    # if result_time == dt.time(hour=00,minute=00,second=00):
    #     result_day = result.date()-dt.timedelta(days=1)
    #     result_time_str = ' 24:00:00'
    #     result_date_str = result_day.strftime("%Y-%m-%d")
    #     result_str = result_date_str+result_time_str
    #     return result_str
    result_str = result.strftime("%Y-%m-%d %H:%M:%S")
    return result_str


# 获取时间sum个15min前的时间序列字符串
def get_time_backward_15min_sequence(time_str, sum):
    time_seq = []
    processed_time_str = time_str
    time_seq.append(processed_time_str)
    for i in range(sum - 1):
        processed_time_str = get_time_backward_15min_str(processed_time_str)
        time_seq.append(processed_time_str)
    return time_seq


# 获取时间后15min的时间字符串
def get_time_forward_15min_str(time_str):
    time_gap = dt.timedelta(minutes=15)
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    result = origin_time + time_gap
    # result_time = result.time()
    # if result_time == dt.time(hour=00,minute=00,second=00):
    #     result_day = result.date() - dt.timedelta(days=1)
    #     result_time_str = ' 24:00:00'
    #     result_date_str = result_day.strftime("%Y-%m-%d")
    #     result_str = result_date_str+result_time_str
    #     return result_str
    result_str = result.strftime("%Y-%m-%d %H:%M:%S")
    return result_str


# 获取时间后15min字符串序列
def get_time_forward_15min_sequence(time_str, sum):
    time_seq = []
    processed_time_str = time_str
    time_seq.append(processed_time_str)
    for i in range(sum - 1):
        processed_time_str = get_time_forward_15min_str(processed_time_str)
        time_seq.append(processed_time_str)
    return time_seq


# 获取时间后1H的时间字符串
def get_time_forward_hour_str(time_str):
    time_gap = dt.timedelta(minutes=60)
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    result = origin_time + time_gap
    result_str = result.strftime("%Y-%m-%d %H:%M:%S")
    return result_str


# 获取时间后sum个1H的时间字符串序列
def get_time_forward_hour_sequence(time_str, sum):
    time_seq = []
    processed_time_str = time_str
    time_seq.append(processed_time_str)
    for i in range(sum - 1):
        processed_time_str = get_time_forward_hour_str(processed_time_str)
        time_seq.append(processed_time_str)
    return time_seq


# 获取时间前1H的时间字符串
def get_time_backward_hour_str(time_str):
    time_gap = dt.timedelta(minutes=60)
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    result = origin_time - time_gap
    result_str = result.strftime("%Y-%m-%d %H:%M:%S")
    return result_str


# 获取时间前sum个1H的时间字符串序列
def get_time_backward_hour_sequence(time_str, sum):
    time_seq = []
    processed_time_str = time_str
    time_seq.append(processed_time_str)
    for i in range(sum - 1):
        processed_time_str = get_time_backward_hour_str(processed_time_str)
        time_seq.append(processed_time_str)
    return time_seq


# 获取时间后1d的时间字符串
def get_time_forward_day_str(time_str):
    time_gap = dt.timedelta(days=1)
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    result = origin_time + time_gap
    result_str = result.strftime("%Y-%m-%d %H:%M:%S")
    return result_str


# 获取时间前sum个1D的时间字符串序列
def get_time_forward_day_sequence(time_str, sum):
    time_seq = []
    processed_time_str = time_str
    time_seq.append(processed_time_str)
    for i in range(sum - 1):
        processed_time_str = get_time_forward_day_str(processed_time_str)
        time_seq.append(processed_time_str)
    return time_seq

# 获取时间后30min的时间字符串
def get_time_forward_30min_str(time_str):
    time_gap = dt.timedelta(minutes=30)
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    result = origin_time + time_gap
    result_str = result.strftime("%Y-%m-%d %H:%M:%S")
    return result_str


# 获取时间前sum个30min的时间字符串序列
def get_time_forward_30min_sequence(time_str, sum):
    time_seq = []
    processed_time_str = time_str
    time_seq.append(processed_time_str)
    for i in range(sum - 1):
        processed_time_str = get_time_forward_30min_str(processed_time_str)
        time_seq.append(processed_time_str)
    return time_seq

# 获取时间序列点
def get_time_point_from_time_str(time_str, start_time_str, time_delta_factor):
    # time_delta_factor是指每个时间点之间有多少分钟,int
    point_sum = 1440 / time_delta_factor
    result_time = get_raw_time_date_from_time_str(time_str)
    start_time = get_raw_time_date_from_time_str(start_time_str)
    time_delta = dt.timedelta(minutes=int(time_delta_factor))
    if int((result_time - start_time) / time_delta) < point_sum:
        time_point = int((result_time - start_time) / time_delta)
    else:
        time_point = int((result_time - start_time) / time_delta) % point_sum
    return time_point


# 从字符串获取时间（hour,minute）
def get_time_from_time_str(time_str):
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    result_time = origin_time.time()
    hour = result_time.hour
    minute = result_time.minute
    return hour, minute


# 从字符串获取时间
def get_raw_time_date_from_time_str(time_str):
    return dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")


# 从字符串获取日期
def get_day_from_time_str(time_str):
    # if(time_str==)
    origin_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    result_day = origin_time.date()
    return result_day


if __name__ == "__main__":
    day_start_time = get_day_start_time_str('2021-12-14 15:45:00')
    print(day_start_time)
    print(get_time_forward_day_sequence('2021-12-14 23:45:00',2))
