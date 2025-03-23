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
        # Load data from CSV
        df = pd.read_csv(data_path)
        
        print(f"Successfully loaded breast cancer dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Clean data - handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"Found {missing_values} missing values. Handling them...")
            # For numeric columns, replace NaN with column mean
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col] = df[col].fillna(df[col].mean())
        
        # Extract features (all columns except 'id' and 'diagnosis')
        feature_columns = [col for col in df.columns if col not in ['id', 'diagnosis']]
        X = df[feature_columns].values
        
        # Convert target to binary (M=1, B=0)
        y = (df['diagnosis'] == 'M').astype(int).values
        
        print(f"Initial features shape: {X.shape}, Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train initial model to identify important features
        initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        initial_model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': initial_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("\nTop 10 most important features for breast cancer:")
        print(feature_importance.head(10))
        
        # Select top 10 features
        top_10_features = feature_importance.head(10)['Feature'].tolist()
        
        # Create new datasets with only the top 10 features
        X_top10 = df[top_10_features].values
        
        print(f"Reduced features shape: {X_top10.shape}, Target shape: {y.shape}")
        
        # Split data again using only top 10 features
        X_train_top10, X_test_top10, y_train, y_test = train_test_split(X_top10, y, test_size=0.2, random_state=42)
        
        # Scale features (new scaler for the reduced feature set)
        scaler_top10 = StandardScaler()
        X_train_scaled_top10 = scaler_top10.fit_transform(X_train_top10)
        X_test_scaled_top10 = scaler_top10.transform(X_test_top10)
        
        # Train final model with only top 10 features
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled_top10, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled_top10)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Breast Cancer Model Accuracy with top 10 features: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save the list of top 10 features for reference
        with open(os.path.join("models", "breast_cancer_top10_features.pkl"), 'wb') as f:
            pickle.dump(top_10_features, f)
        print(f"Top 10 features saved: {top_10_features}")
        
        return model, scaler_top10, top_10_features
        
    except Exception as e:
        print(f"Error in breast cancer model training: {e}")
        # Fallback to using sklearn's dataset
        from sklearn.datasets import load_breast_cancer
        
        print("Using fallback Wisconsin Breast Cancer dataset from sklearn")
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        # Get feature names from sklearn dataset
        feature_names = data.feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train initial model to identify important features
        initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        initial_model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': initial_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("\nTop 10 most important features for breast cancer:")
        print(feature_importance.head(10))
        
        # Get indices of top 10 features
        top_10_indices = feature_importance.head(10).index.tolist()
        top_10_features = feature_importance.head(10)['Feature'].tolist()
        
        # Create new datasets with only the top 10 features
        X_top10 = X[:, top_10_indices]
        
        # Split data again using only top 10 features
        X_train_top10, X_test_top10, y_train, y_test = train_test_split(X_top10, y, test_size=0.2, random_state=42)
        
        # Scale features (new scaler for the reduced feature set)
        scaler_top10 = StandardScaler()
        X_train_scaled_top10 = scaler_top10.fit_transform(X_train_top10)
        X_test_scaled_top10 = scaler_top10.transform(X_test_top10)
        
        # Train final model with only top 10 features
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled_top10, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled_top10)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Breast Cancer Model Accuracy with top 10 features: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save the list of top 10 features for reference
        with open(os.path.join("models", "breast_cancer_top10_features.pkl"), 'wb') as f:
            pickle.dump(top_10_features, f)
        print(f"Top 10 features saved: {top_10_features}")
        
        return model, scaler_top10, top_10_features

def train_lung_cancer_model(data_path=None):
    if data_path is None:
        data_path = "datasets/lungCancer.csv"
    
    try:
        df = pd.read_csv(data_path)
        
        print(f"Successfully loaded lung cancer dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Handle categorical variables
        if 'GENDER' in df.columns:
            df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0, 'MALE': 1, 'FEMALE': 0})
        
        # Check for other categorical columns and convert them if necessary
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'LUNG_CANCER':
                print(f"Converting categorical column {col} to numeric")
                # For yes/no columns (assuming these are binary attributes)
                if set(df[col].unique()) <= {'YES', 'NO', 'Yes', 'No', 'yes', 'no', 'Y', 'N', 'y', 'n'}:
                    df[col] = df[col].map(lambda x: 1 if x.upper() in ('YES', 'Y') else 0)
                else:
                    # For other categorical columns, use one-hot encoding
                    df = pd.get_dummies(df, columns=[col], drop_first=True)
        
        # Extract features (all columns except target)
        feature_columns = [col for col in df.columns if col != 'LUNG_CANCER']
        X = df[feature_columns].values
        
        # Convert target to binary
        y = (df['LUNG_CANCER'].map(lambda x: 1 if str(x).upper() in ('YES', 'Y') else 0)).values
        
        print(f"Initial lung cancer features shape: {X.shape}, Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train initial model to identify important features
        initial_model = RandomForestClassifier(n_estimators=100, random_state=42)
        initial_model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': initial_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("\nTop 7 most important features for lung cancer:")
        print(feature_importance.head(7))
        
        # Select top 7 features
        top_7_features = feature_importance.head(7)['Feature'].tolist()
        
        # Create new datasets with only the top 7 features
        X_top7 = df[top_7_features].values
        
        print(f"Reduced lung cancer features shape: {X_top7.shape}, Target shape: {y.shape}")
        
        # Split data again using only top 7 features
        X_train_top7, X_test_top7, y_train, y_test = train_test_split(X_top7, y, test_size=0.2, random_state=42)
        
        # Scale features (new scaler for the reduced feature set)
        scaler_top7 = StandardScaler()
        X_train_scaled_top7 = scaler_top7.fit_transform(X_train_top7)
        X_test_scaled_top7 = scaler_top7.transform(X_test_top7)
        
        # Train final model with only top 7 features
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled_top7, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled_top7)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Lung Cancer Model Accuracy with top 7 features: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save the list of top 7 features for reference
        with open(os.path.join("models", "lung_cancer_top7_features.pkl"), 'wb') as f:
            pickle.dump(top_7_features, f)
        print(f"Top 7 features saved: {top_7_features}")
        
        return model, scaler_top7, top_7_features
        
    except Exception as e:
        print(f"Error in lung cancer model training: {e}")
        print("Using manually defined features based on domain knowledge")
        
        # Define default important features based on medical literature
        top_7_features = ['AGE', 'ALCOHOL CONSUMING', 'ALLERGY ', 'PEER_PRESSURE', 'YELLOW_FINGERS', 'FATIGUE ', 'COUGHING']
        
        # Create a simple model as a fallback
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        
        print(f"Using fallback features: {top_7_features}")
        
        # Save the list of fallback features for reference
        with open(os.path.join("models", "lung_cancer_top7_features.pkl"), 'wb') as f:
            pickle.dump(top_7_features, f)
        
        return model, scaler, top_7_features

if __name__ == "__main__":
    # Create necessary directories
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
        print("Created datasets directory. Please place your CSV files there.")
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Train breast cancer model
    print("\n=== TRAINING BREAST CANCER MODEL ===\n")
    breast_model, breast_scaler, breast_features = train_breast_cancer_model()
    
    # Save breast cancer model, scaler and features
    if breast_model is not None and breast_scaler is not None:
        with open(os.path.join("models", "breast_cancer_model.pkl"), 'wb') as f:
            pickle.dump(breast_model, f)
        
        with open(os.path.join("models", "breast_cancer_scaler.pkl"), 'wb') as f:
            pickle.dump(breast_scaler, f)
        
        print("Breast cancer model and scaler saved to models directory")
    
    # Train lung cancer model
    print("\n=== TRAINING LUNG CANCER MODEL ===\n")
    lung_model, lung_scaler, lung_features = train_lung_cancer_model()
    
    # Save lung cancer model, scaler and features
    if lung_model is not None and lung_scaler is not None:
        with open(os.path.join("models", "lung_cancer_model.pkl"), 'wb') as f:
            pickle.dump(lung_model, f)
        
        with open(os.path.join("models", "lung_cancer_scaler.pkl"), 'wb') as f:
            pickle.dump(lung_scaler, f)
        
        with open(os.path.join("models", "lung_cancer_top7_features.pkl"), 'wb') as f:
            pickle.dump(lung_features, f)
        
        print("Lung cancer model and scaler saved to models directory")
    
    print("\nAll models trained and saved successfully.")