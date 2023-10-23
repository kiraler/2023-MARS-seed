import numpy as np
import pandas as pd


def get_date():
    from workalendar.asia import China
    cal = China()
    df_holiday = pd.DataFrame()
    for year in [2022, 2023]:
        holidays = cal.holidays(year)
        df_tmp = pd.DataFrame(holidays, columns=['time', 'Holiday'])
        df_tmp['Holiday Length'] = df_tmp.groupby('Holiday')['time'].transform(lambda x: x.count())
        # df_tmp1 = df_tmp[df_tmp['Holiday'] == 'Spring Festival'].reset_index(drop=True).reset_index().rename(columns={'index': 'day_num'})
        # df_tmp = df_tmp.merge(df_tmp1[['time','day_num']], on='time', how='left')
        df_holiday = pd.concat([df_holiday, df_tmp])
    df_holiday['time'] = pd.to_datetime(df_holiday['time'])
    # df_holiday['Festival'] = df_holiday['Holiday'].apply(lambda x: 1 if x=='Spring Festival' else 0)
    # 时间补全
    df_time_all = pd.DataFrame()
    for i in range(500):
        df_time = pd.DataFrame()
        df_time['time'] = pd.date_range(start='20220415', end='20230421', freq='D')
        df_time['id_encode'] = i
        df_time_all = pd.concat([df_time_all, df_time])
    
    df_time_all = df_time_all.merge(df_holiday, on='time',how='left')
    df_time_all['Holiday'] = df_time_all['Holiday'].apply(lambda x: 0 if x is np.nan else 1)
    df_time_all['Holiday Length'] = df_time_all['Holiday Length'].fillna(0)
    print('df_time_all.shape:', df_time_all.shape)
    return df_time_all


def city_preprocess(df_city, df_weather):
    df_weather['time'] = pd.to_datetime(df_weather['date'])
    df_city = df_city[['h3', 'longitude', 'latitude', 'city']]
    df_city = df_city.merge(df_weather[['city', 'time', 'temp_max', 'temp_min', 'weather']], on='city')

    # df_weather['weather'].value_counts()
    df_city['temp_max'] = df_city['temp_max'].astype(int)
    df_city['temp_min'] = df_city['temp_min'].astype(int)
    df_city['temp_mean'] = df_city[['temp_max','temp_min']].mean(axis=1)
    df_city['temp_diff'] = df_city['temp_max'] - df_city['temp_min']
    df_city['weather_status'] = df_city['weather'].apply(lambda x: 1 if '雨' in x 
                                                                else 2 if '晴' in x 
                                                                else -1 if '雾' in x
                                                                else 0)
    return df_city


def data_day_group(df_his_all, df_power):
    """数据天聚合"""
    df_his_day = df_his_all.groupby(['id_encode','ds']).agg(['mean','std']).reset_index()
    df_his_day.columns = [x[0] if x[1] == '' else f'{x[0]}_{x[1]}' for x in df_his_day.columns]
    df_his_day['all_price'] = df_his_day['ele_price_mean'] + df_his_day['after_ser_price_mean']
    df_his_day['折扣'] = df_his_day['after_ser_price_mean'] / (df_his_day['ser_price_mean']+0.01)
    df_his_day = df_his_day.drop(['hour_mean', 'hour_std', 'f3_std',], axis=1)
    # df_his_day['f2/f1'] = df_his_day['f2_mean'] / (df_his_day['f1_mean'] + 0.01)

    df_power_day = df_power.groupby(['id_encode','ds']).agg({'power':['sum','std']}).reset_index()
    df_power_day.columns = [x[0] if x[1] == '' else f'{x[0]}_{x[1]}' for x in df_power_day.columns]
    df_power_mean = df_power.groupby('id_encode')['power'].mean().reset_index()
    return df_his_day, df_power_day, df_power_mean


def data_merge(df_stub, df_his_day, df_power_day, df_power_mean, df_city):
    """数据拼接"""
    df_time_all = get_date()
    df = df_his_day.merge(df_power_day, on=['id_encode','ds'],how='left')
    df = df.merge(df_power_mean, on=['id_encode'],how='left')
    df = df.merge(df_stub, on='id_encode', how='left')
    df['time'] = pd.to_datetime(df['ds'], format='%Y%m%d')
    df = df_time_all.merge(df, how='left', on=['time', 'id_encode'])
    df['ds'] = df['ds'].fillna(0).astype(int)
    df['is_holiday'] = df['Holiday'] #df['ds'].apply(lambda x: 1 if x in holidays_list else 0)
    return df


def feat_engineering(df: pd.DataFrame):
    print('ori df:', df.shape)
    df['flag'], unique_categories = pd.factorize(df['flag'])
    # 标签变换
    # df['label'] = np.log(df['power_sum']+1)
    # df['label'] = df['power_sum'] / (df['power']+1)
    df['label'] = df['power_sum']
    # 时序特征
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['month_num'] = df['month'] + 12*(df['year']-df['year'].min())
    df['weeks'] = ((df['time'] - df['time'].min()).dt.days + 3) // 7
    df['day_of_week'] = df['time'].dt.dayofweek
    # df['quarter'] = df['time'].dt.quarter
    
    for i in range(1, 5):
        # ok --testA: 235.8-> 229.8 oof 265.9 -> 261.2
        df[f'power_{i}week'] = df.groupby('id_encode')['power_sum'].apply(lambda x: x.shift(7 * i))
        # oof rmse:  261.2 -> 264.3没用
        # df[f'power_{i}week_daystd'] = df.groupby('id_encode')['power_std'].apply(lambda x: x.shift(7 * i))

    # # oof rmse:  261.2 -> 268.3没用
    # for i in range(1, 4):
    #     df[f'power_{i}week_diff'] = df[f'power_{i}week'] - df[f'power_{i+1}week']

    for i in [7, 14, 21, 30]:
        df[f'power_roll{i}_{i}ago'] = df.groupby('id_encode')['label'].transform(lambda x: x.rolling(window=i).mean().shift(i))
        df[f'power_roll{i}_{i}ago_std'] = df.groupby('id_encode')['label'].transform(lambda x: x.rolling(window=i).std().shift(i))

        # df[f'power_{i}ago'] = df.groupby('id_encode')['label'].transform(lambda x: x.shift(i))
        # 没什么用
        # df[f'power_roll3_{i}ago'] = df.groupby('id_encode')['power_sum'].transform(lambda x: x.rolling(window=3).mean().shift(i))
        # df[f'power_roll3_{i}ago_std'] = df.groupby('id_encode')['power_sum'].transform(lambda x: x.rolling(window=3).std().shift(i))

        # df[f'power_roll{i}_{i}ago_diff7'] = df.groupby('id_encode')['label'].transform(lambda x: x.rolling(window=i).mean().shift(i) - x.rolling(window=i).mean().shift(i+7))
        # if i != 7:
        #     df[f'power_roll7_{i}ago'] = df.groupby('id_encode')['power_sum'].transform(lambda x: x.rolling(window=7).mean().shift(i))
        #     df[f'power_roll7_{i}ago_std'] = df.groupby('id_encode')['power_sum'].transform(lambda x: x.rolling(window=7).std().shift(i))

    # 两周前统计
    df_weeks = df.groupby(['id_encode', 'weeks']).agg({'label': ['mean', 'std']}).reset_index()
    # for i in [2]:
    df_weeks.columns = [x[0] if x[1] == '' else f'{x[0]}_{x[1]}_2week' for x in df_weeks.columns]
    df_weeks['weeks'] = df_weeks.groupby('id_encode')['weeks'].apply(lambda x: x.shift(-2))
    df = df.merge(df_weeks, on=['id_encode', 'weeks'], how='left')

    # # 两周前全局统计，过拟合
    # df_weeks = df.groupby(['weeks']).agg({'power_sum': ['mean', 'std']}).reset_index()
    # df_weeks.columns = [x[0] if x[1] == '' else f'{x[0]}_{x[1]}_2week_all' for x in df_weeks.columns]
    # df_weeks['weeks'] = df_weeks['weeks'].shift(-2)
    # df = df.merge(df_weeks, on=['weeks'], how='left')

    # 上月统计
    col = 'month_num'
    df_weeks = df.groupby(['id_encode', col]).agg({'label': ['mean', 'std']}).reset_index()
    df_weeks.columns = [x[0] if x[1] == '' else f'{x[0]}_{x[1]}_{col}_2' for x in df_weeks.columns]
    df_weeks[col] = df_weeks.groupby('id_encode')[col].apply(lambda x: x.shift(-2))
    df = df.merge(df_weeks, on=['id_encode', col], how='left')
    
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # df['h3'] = le.fit_transform(df['h3'])
    # testA 229.8 -> 231.0 oof 261->260不一致
    # for col in ['f1_mean', 'f2_mean', 'f3_mean']:
    #     i = 7
    #     df[f'{col}_roll{i}'] = df.groupby('id_encode')[col].transform(lambda x: x.rolling(window=i).mean())
    #     df[f'{col}_roll{i}_std'] = df.groupby('id_encode')[col].transform(lambda x: x.rolling(window=i).std())
    # for col in ['f1_mean', 'f2_mean', 'f3_mean']:
    #     for i in [1, 7, 15]:
    #         df[f'{col}_diff{i}'] = df.groupby('id_encode')[col].transform(lambda x: x - x.shift(-i))
    print('feat df:', df.shape)
    return df
