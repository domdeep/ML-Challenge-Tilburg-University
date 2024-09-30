import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
from pandas.tseries.holiday import USFederalHolidayCalendar
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from zipfile import ZipFile


def load_data(train_path, test_path):
    """
    Load training and testing data from CSV files.
    """
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        exit(1)
    except pd.errors.EmptyDataError as e:
        print(f"Invalid CSV file: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

def preprocess_data(train, test):
    """
    Preprocess the data by removing negative values, handling outliers, and imputing missing values.
    """
    # Remove negative values in CurrentSessionLength
    train = train[train['CurrentSessionLength'] >= 0]
    
    # Removing outliers using IQR filtering in CurrentSessionLength
    Q1 = train['CurrentSessionLength'].quantile(0.25)
    Q3 = train['CurrentSessionLength'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train = train[(train['CurrentSessionLength'] >= lower_bound) & (train['CurrentSessionLength'] <= upper_bound)]

    # Median imputation for LevelProgressionAmount
    median_value = train['LevelProgressionAmount'].median()
    train['LevelProgressionAmount'].fillna(median_value, inplace=True)
    median_value = test['LevelProgressionAmount'].median()
    test['LevelProgressionAmount'].fillna(median_value, inplace=True)

    return train, test

def feature_engineering(train, test):
    """
    Apply feature engineering to both training and test datasets.
    """
    # Convert TimeUtc to datetime and create related features
    for df in [train, test]:
        df['TimeUtc'] = pd.to_datetime(df['TimeUtc'])
        df['Second'] = df['TimeUtc'].dt.second
        df['Minute'] = df['TimeUtc'].dt.minute
        df['Hour'] = df['TimeUtc'].dt.hour
        df['Day'] = df['TimeUtc'].dt.day
        df['DayOfWeek'] = df['TimeUtc'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'] >= 5
        df['Month'] = df['TimeUtc'].dt.month
        df['Quarter'] = df['TimeUtc'].dt.quarter
        df['ElapsedTime'] = (df['TimeUtc'] - df['TimeUtc'].min()).dt.total_seconds()
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

def encode_features(train, test):
    """
    Apply frequency and target encoding to user IDs.
    """
    frequency_encoding = train['UserID'].value_counts(normalize=True).to_dict()
    train['UserID_encoded'] = train['UserID'].map(frequency_encoding)
    test['UserID_encoded'] = test['UserID'].map(frequency_encoding)
    test['UserID_encoded'].fillna(0, inplace=True)

    encoder = TargetEncoder(smoothing=0.7)
    train['UserID_targetencoded'] = encoder.fit_transform(train['UserID'], train['ResponseValue'])
    test['UserID_targetencoded'] = encoder.transform(test['UserID'])
    global_mean = train['ResponseValue'].mean()
    test['UserID_targetencoded'].fillna(global_mean, inplace=True)

def drop_unnecessary_columns(train, test):
    """
    Drop columns that are not needed for modeling.
    """
    columns_to_drop = [
        'TimeUtc', 'QuestionTiming', 'CurrentGameMode',
        'CurrentTask', 'LastTaskCompleted',
        'LastTaskCompleted_Aggregated', 'QuestionType', 'UserID',
    ]
    train.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    test.drop(columns=columns_to_drop, inplace=True, errors='ignore')

def train_and_evaluate(train):
    """
    Train the RandomForestRegressor and evaluate its performance.
    """
    X = train.drop(columns=['ResponseValue'])
    y = train['ResponseValue']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42
    )
    rf.fit(X_train, y_train)
    val_predictions = rf.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    print("Mean Absolute Error on Validation Set:", val_mae)
    return rf, X.columns

def save_predictions(model, test):
    """
    Make predictions on test data and save the results.
    """
    test_predictions = model.predict(test)
    pd.DataFrame(test_predictions).to_csv('predicted.csv', index=False, header=False)
    print(f"Predictions saved to 'predicted.csv'.")

def create_zip_file():
    """
    Create a zip file containing the predictions.
    """
    with ZipFile('predictions.zip', 'w') as zipf:
        zipf.write('predicted.csv', arcname='predicted.csv')
    print("Zip file created with the predictions.")

if __name__ == "__main__":
    train, test = load_data("train_data.csv", "test_data.csv")
    train, test = preprocess_data(train, test)
    feature_engineering(train, test)
    encode_features(train, test)
    drop_unnecessary_columns(train, test)
    rf, feature_columns = train_and_evaluate(train)
    save_predictions(rf, test)
    create_zip_file() 