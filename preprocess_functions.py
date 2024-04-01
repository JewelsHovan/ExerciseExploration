# Preprocessing
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import missingno as msno


def preprocess_strava_data(data, time_threshold=30):
    """
    Preprocess the input data frame by converting timestamps to datetime,
    extracting date and time components, forward-filling missing values in
    specified columns, and splitting the data into running and cycling
    segments based on a specific date.

    Parameters:
    data (pd.DataFrame): The input data frame containing the activity data,
                         including a 'timestamp' column and others like
                         'cadence', 'enhanced_altitude', etc.

    Returns:
    tuple: A tuple containing two data frames, (running_df, cycling_df), where
           running_df includes activities before September 12, 2019, and
           cycling_df includes activities on or after September 12, 2019.
    """
    # Convert 'timestamp' column to datetime and extract 'date' and 'time'
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['date'] = data['timestamp'].dt.date
    data['time'] = data['timestamp'].dt.time

    # replace infinite values with nan
    data = data.replace([np.inf, -np.inf], np.nan) # Replace infinities with NaN

    # Forward fill missing values in specified columns
    cols_with_few_missing = ['cadence', 'enhanced_altitude', 'enhanced_speed', 'fractional_cadence', 'position_lat', 'position_long']
    data[cols_with_few_missing] = data[cols_with_few_missing].ffill()
    
    # remove datafile column
    cols_to_drop = ['Air Power', 'Cadence', 'Form Power', 'Ground Time', 'Leg Spring Stiffness', 'Power', 'Vertical Oscillation', 'altitude','datafile','unknown_87', 'unknown_88', 'unknown_90']
    data.drop(cols_to_drop, axis=1, inplace=True)
    
    # Split data into running and cycling based on the date cutoff
    running_data = data[data['date'] < datetime.date(2019, 9, 12)]
    cycling_data = data[data['date'] >= datetime.date(2019, 9, 12)]
    cycling_data = cycling_data.loc[:, 'cadence':]
    cycling_data.drop('speed', axis=1, inplace=True)

    # add activity period -> segregate data by activity periods (a duration of exercise)
    # Define the threshold for a new activity period
    threshold = pd.Timedelta(minutes=time_threshold)

    # Identify separate activity periods
    running_data['time_diff'] = running_data['timestamp'].diff()
    cycling_data['time_diff'] = cycling_data['timestamp'].diff()

    running_data['activity_period'] = (running_data['time_diff'] > threshold).cumsum()
    cycling_data['activity_period'] = (cycling_data['time_diff'] > threshold).cumsum()

    return running_data, cycling_data

def process_activity_data(df, max_cols, mean_cols, time_threshold=30):
    # Add a 'time_diff' column with the difference between each timestamp
    df['time_diff'] = df['timestamp'].diff()

    # Define the threshold for a new activity period
    threshold = pd.Timedelta(minutes=time_threshold)

    # Identify separate activity periods
    df['activity_period'] = (df['time_diff'] > threshold).cumsum()

    # Group by 'activity_period' and calculate the start and end timestamps
    activity_periods = df.groupby('activity_period')['timestamp'].agg(['max', 'min'])

    # Calculate the time length of each activity period in minutes
    activity_periods['time_length'] = (activity_periods['max'] - activity_periods['min']).dt.total_seconds() / 60

    # Define the aggregation functions for specified columns
    agg_funcs = {col: ['max', 'mean'] for col in ['heart_rate', 'cadence', 'enhanced_altitude'] + max_cols + mean_cols}

    # Aggregate the data by 'activity_period' using the specified functions
    activity_by_period = df.groupby(['date', 'activity_period']).agg(agg_funcs).reset_index()

    # Flatten the multi-level column names
    activity_by_period.columns = ['_'.join(col).rstrip('_') for col in activity_by_period.columns.values]

    # Add the 'time_length' to the aggregated data
    activity_by_period = activity_by_period.merge(activity_periods['time_length'], left_on='activity_period', right_index=True)

    # Calculate distance per minute for each activity period
    if 'distance_max' in activity_by_period.columns:
        activity_by_period['distance_per_min'] = activity_by_period['distance_max'] / activity_by_period['time_length']

    return activity_by_period
