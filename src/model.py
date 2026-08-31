import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_baggage_risk_model(df: pd.DataFrame):
    #Trains a Random Forest classifier to predict baggage delay risk.
    features = ['MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'DEP_DELAY', 'connect_time_mins', 'baggage_count']
    X = df[features]
    y = df['baggage_misconnect']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train.values, y_train)
    
    return model, features

def predict_risk(model, features_dict: dict) -> float:
    #Predicts probability of baggage misconnect given input parameters.
    feature_order = ['MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'DEP_DELAY', 'connect_time_mins', 'baggage_count']
    
    input_data = [[features_dict[feat] for feat in feature_order]]
    probabilities = model.predict_proba(input_data)
    
    if probabilities.shape[1] > 1:
        return float(probabilities[0][1])
    return 0.0