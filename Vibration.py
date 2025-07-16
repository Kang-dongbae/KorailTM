import pandas as pd
import numpy as np

# 시간에서 초 단위 추출
def parse_time_to_seconds(time_str):
    try:
        return pd.to_timedelta(time_str).total_seconds()
    except (ValueError, TypeError):
        return float(time_str)

def vibration_time_feature(df, sampling_rate):

    # 초 시간 분리
    df['time_seconds'] = df['time'].apply(parse_time_to_seconds)
    data = df['signal'].values
    time_seconds = df['time_seconds'].values
    N = len(data)
    duration = time_seconds[-1] - time_seconds[0]
    
    # 초 단위 구간 정의 (벡터화 처리)
    min_time, max_time = int(time_seconds[0]), int(time_seconds[-1]) + 1
    time_bins = np.arange(min_time, max_time, 1)
    bin_indices = np.digitize(time_seconds, time_bins) - 1

    # 평균, RMS, Square root amplitude, peak to peak,   
    results = []
    for i in range(len(time_bins) - 1):
        mask = (bin_indices == i)
        if not np.any(mask):
            continue
        segment = data[mask]
        
        # 시간 도메인 파라미터 (벡터화 계산)
        mean_val = np.mean(segment)
        rms_val = np.sqrt(np.mean(segment**2))

    # 시간 값 (구간 시작 시간)
    segment_time = df['time'].iloc[np.where(mask)[0][0]]
    
    # 결과
    results.append({
        'time': segment_time,
        'mean': mean_val,
        'rms': rms_val
    }) 

    return pd.DataFrame(results)


# vibData = pd.read_csv('Test.csv')

