import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pickle

def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(data):
    data.fillna(method='ffill', inplace=True)
    return data

def encode_categorical_features(data):
    return pd.get_dummies(data, drop_first=True)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def add_features(data):
    data['car_age'] = 2024 - data['year']
    data.drop('year', axis=1, inplace=True)
    return data

def preprocess_data(data):
    data = handle_missing_values(data)
    data = encode_categorical_features(data)
    data = add_features(data)
    return data

# Load and preprocess the dataset
data = load_data('data/used_car_data.csv')
data = preprocess_data(data)

# Split the dataset into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
X_train, X_test, scaler = scale_features(X_train, X_test)

# Train models
def train_models(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    
    return lr_model, dt_model

# Evaluate models
def evaluate_models(models, X_test, y_test):
    results = {}
    for model in models:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        results[model.__class__.__name__] = mae
    return results

if __name__ == "__main__":
    lr_model, dt_model = train_models(X_train, y_train)
    models = [lr_model, dt_model]
    results = evaluate_models(models, X_test, y_test)
    
    for model, mae in results.items():
        print(f"{model} MAE: {mae}")
    
    best_model = dt_model if results['DecisionTreeRegressor'] < results['LinearRegression'] else lr_model
    pickle.dump(best_model, open('models/best_model.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
