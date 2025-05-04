from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)

## Step 1: Data Loading and Exploration
def load_data():
    train = pd.read_csv("https://raw.githubusercontent.com/madhav481010/Inventory-Demand-Forecasting/main/train.csv", names=columns, header=0)
    test = pd.read_csv("https://raw.githubusercontent.com/madhav481010/Inventory-Demand-Forecasting/main/test.csv", names=columns, header=0)
    stores = pd.read_csv("https://raw.githubusercontent.com/madhav481010/Inventory-Demand-Forecastings/main/stores.csv", names=columns, header=0)
    features = pd.read_csv("https://raw.githubusercontent.com/madhav481010/Inventory-Demand-Forecasting/main/features.csv", names=columns, header=0)

    train = pd.merge(train, features, on=['Store', 'Date'], how='left')
    train = pd.merge(train, stores, on=['Store'], how='left')

    test = pd.merge(test, features, on=['Store', 'Date'], how='left')
    test = pd.merge(test, stores, on=['Store'], how='left')

    return train, test

train_df, test_df = load_data()

print("\nTrain Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("\nTrain Data Columns:", train_df.columns.tolist())


## Step 2: Data Preprocessing
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day

    holidays = {
        'SuperBowl': ['2010-02-12', '2011-02-11', '2012-02-10'],
        'LaborDay': ['2010-09-10', '2011-09-09', '2012-09-07'],
        'Thanksgiving': ['2010-11-26', '2011-11-25', '2012-11-23'],
        'Christmas': ['2010-12-31', '2011-12-30', '2012-12-28']
    }

    for holiday, dates in holidays.items():
        df[holiday] = df['Date'].isin(pd.to_datetime(dates)).astype(int)

    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    df['CPI'].fillna(df['CPI'].mean(), inplace=True)
    df['Unemployment'].fillna(df['Unemployment'].mean(), inplace=True)

    min_date = df['Date'].min()
    df['Days_Since_Start'] = (df['Date'] - min_date).dt.days

    df['Size_to_Type'] = df['Size'] / df.groupby('Type')['Size'].transform('mean')

    return df

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

features = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price',
            'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
            'CPI', 'Unemployment', 'Size', 'Type', 'Year', 'Month', 'Week',
            'Day', 'SuperBowl', 'LaborDay', 'Thanksgiving', 'Christmas',
            'Days_Since_Start', 'Size_to_Type']
target = 'Weekly_Sales'

if 'IsHoliday' in train_df.columns:
    features = [f if f != 'IsHoliday' else 'IsHoliday' for f in features]
elif 'isholiday' in train_df.columns:
    features = [f if f != 'IsHoliday' else 'isholiday' for f in features]
else:
    features.remove('IsHoliday')  
  
X = train_df[features]
y = train_df[target]

X_test_final = test_df[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


## Step 3: Feature Engineering Pipeline
categorical_features = ['Store', 'Dept', 'Type', 'IsHoliday',
                       'SuperBowl', 'LaborDay', 'Thanksgiving', 'Christmas']

if 'IsHoliday' not in features:
    categorical_features.remove('IsHoliday')
numerical_features = [f for f in features if f not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=25, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=25, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=25, random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'model': pipeline
    }



## Step 5: Model Comparison and Visualization
print("\nModel Comparison:")
print("{:<20} {:<10} {:<10} {:<10}".format('Model', 'MAE', 'RMSE', 'R2'))
for name, metrics in results.items():
    print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f}".format(
        name, metrics['MAE'], metrics['RMSE'], metrics['R2']))


## Step 6: Final Model Selection and Prediction

best_model_name = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")

print("\nTraining best model on full data...")
best_model.fit(X, y)

print("\nMaking predictions on test data...")
final_predictions = best_model.predict(X_test_final)

submission = test_df[['Store', 'Dept', 'Date']].copy()
submission['Weekly_Sales'] = final_predictions
submission.to_csv('walmart_sales_forecast_submission.csv', index=False)
print("\nPredictions saved to 'walmart_sales_forecast_submission.csv'")

print("\nBest Model Performance Summary:")
print(f"Model: {best_model_name}")
print(f"Validation MAE: {results[best_model_name]['MAE']:.2f}")
print(f"Validation RMSE: {results[best_model_name]['RMSE']:.2f}")
print(f"Validation R2: {results[best_model_name]['R2']:.2f}")
