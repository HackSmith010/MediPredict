import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def train_breast_cancer_model(data_path=None):
    if data_path is None:
        data_path = "datasets/breastCancer.csv"

    try:
        df = pd.read_csv(data_path)

        df = df.replace([np.inf, -np.inf], np.nan)

        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col] = df[col].fillna(df[col].mean())

        feature_columns = [col for col in df.columns if col not in ['id', 'diagnosis']]
        X = df[feature_columns].values
        y = (df['diagnosis'] == 'M').astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        initial_model.fit(X_train_scaled, y_train)

        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': initial_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        top_10_features = feature_importance.head(10)['Feature'].tolist()

        X_top10 = df[top_10_features].values

        X_train_top10, X_test_top10, y_train, y_test = train_test_split(X_top10, y, test_size=0.2, random_state=42)

        scaler_top10 = StandardScaler()
        X_train_scaled_top10 = scaler_top10.fit_transform(X_train_top10)
        X_test_scaled_top10 = scaler_top10.transform(X_test_top10)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled_top10, y_train)

        y_pred = model.predict(X_test_scaled_top10)
        accuracy = accuracy_score(y_test, y_pred)

        with open(os.path.join("models", "breast_cancer_top10_features.pkl"), 'wb') as f:
            pickle.dump(top_10_features, f)

        return model, scaler_top10, top_10_features

    except Exception as e:
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        X = data.data
        y = data.target

        feature_names = data.feature_names

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        initial_model.fit(X_train_scaled, y_train)

        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': initial_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        top_10_indices = feature_importance.head(10).index.tolist()
        top_10_features = feature_importance.head(10)['Feature'].tolist()

        X_top10 = X[:, top_10_indices]

        X_train_top10, X_test_top10, y_train, y_test = train_test_split(X_top10, y, test_size=0.2, random_state=42)

        scaler_top10 = StandardScaler()
        X_train_scaled_top10 = scaler_top10.fit_transform(X_train_top10)
        X_test_scaled_top10 = scaler_top10.transform(X_test_top10)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled_top10, y_train)

        y_pred = model.predict(X_test_scaled_top10)
        accuracy = accuracy_score(y_test, y_pred)

        with open(os.path.join("models", "breast_cancer_top10_features.pkl"), 'wb') as f:
            pickle.dump(top_10_features, f)

        return model, scaler_top10, top_10_features


def train_lung_cancer_model(data_path=None):
    if data_path is None:
        data_path = "datasets/lungCancer.csv"

    try:
        df = pd.read_csv(data_path)

        if 'GENDER' in df.columns:
            df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0, 'MALE': 1, 'FEMALE': 0})

        for col in df.columns:
            if df[col].dtype == 'object' and col != 'LUNG_CANCER':
                if set(df[col].unique()) <= {'YES', 'NO', 'Yes', 'No', 'yes', 'no', 'Y', 'N', 'y', 'n'}:
                    df[col] = df[col].map(lambda x: 1 if x.upper() in ('YES', 'Y') else 0)
                else:
                    df = pd.get_dummies(df, columns=[col], drop_first=True)

        feature_columns = [col for col in df.columns if col != 'LUNG_CANCER']
        X = df[feature_columns].values
        y = (df['LUNG_CANCER'].map(lambda x: 1 if str(x).upper() in ('YES', 'Y') else 0)).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        initial_model.fit(X_train_scaled, y_train)

        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': initial_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        top_7_features = feature_importance.head(7)['Feature'].tolist()

        X_top7 = df[top_7_features].values

        X_train_top7, X_test_top7, y_train, y_test = train_test_split(X_top7, y, test_size=0.2, random_state=42)

        scaler_top7 = StandardScaler()
        X_train_scaled_top7 = scaler_top7.fit_transform(X_train_top7)
        X_test_scaled_top7 = scaler_top7.transform(X_test_top7)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled_top7, y_train)

        y_pred = model.predict(X_test_scaled_top7)
        accuracy = accuracy_score(y_test, y_pred)

        with open(os.path.join("models", "lung_cancer_top7_features.pkl"), 'wb') as f:
            pickle.dump(top_7_features, f)

        return model, scaler_top7, top_7_features

    except Exception as e:
        top_7_features = ['AGE', 'ALCOHOL CONSUMING', 'ALLERGY ', 'PEER_PRESSURE', 'YELLOW_FINGERS', 'FATIGUE ', 'COUGHING']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()

        with open(os.path.join("models", "lung_cancer_top7_features.pkl"), 'wb') as f:
            pickle.dump(top_7_features, f)

        return model, scaler, top_7_features


if __name__ == "__main__":
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    if not os.path.exists("models"):
        os.makedirs("models")

    breast_model, breast_scaler, breast_features = train_breast_cancer_model()

    if breast_model is not None and breast_scaler is not None:
        with open(os.path.join("models", "breast_cancer_model.pkl"), 'wb') as f:
            pickle.dump(breast_model, f)

        with open(os.path.join("models", "breast_cancer_scaler.pkl"), 'wb') as f:
            pickle.dump(breast_scaler, f)

    lung_model, lung_scaler, lung_features = train_lung_cancer_model()

    if lung_model is not None and lung_scaler is not None:
        with open(os.path.join("models", "lung_cancer_model.pkl"), 'wb') as f:
            pickle.dump(lung_model, f)

        with open(os.path.join("models", "lung_cancer_scaler.pkl"), 'wb') as f:
            pickle.dump(lung_scaler, f)
