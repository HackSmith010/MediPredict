from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BREAST_CANCER_MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_model.pkl")
LUNG_CANCER_MODEL_PATH = os.path.join(MODEL_DIR, "lung_cancer_model.pkl")
BREAST_CANCER_SCALER_PATH = os.path.join(MODEL_DIR, "breast_cancer_scaler.pkl")
LUNG_CANCER_SCALER_PATH = os.path.join(MODEL_DIR, "lung_cancer_scaler.pkl")
BREAST_CANCER_FEATURES_PATH = os.path.join(MODEL_DIR, "breast_cancer_top10_features.pkl")
LUNG_CANCER_FEATURES_PATH = os.path.join(MODEL_DIR, "lung_cancer_top7_features.pkl")

def check_scaler_is_fitted(scaler, name):
    try:
        if hasattr(scaler, 'n_features_in_'):
            sample = np.zeros((1, scaler.n_features_in_))
        else:
            sample = np.zeros((1, 10))  
        
        scaler.transform(sample)
        logger.info(f"{name} scaler is properly fitted")
        return True
    except Exception as e:
        logger.error(f"{name} scaler is not properly fitted: {e}")
        return False

breast_cancer_model = load_model(BREAST_CANCER_MODEL_PATH)
lung_cancer_model = load_model(LUNG_CANCER_MODEL_PATH)
breast_cancer_scaler = load_model(BREAST_CANCER_SCALER_PATH)
lung_cancer_scaler = load_model(LUNG_CANCER_SCALER_PATH)
breast_cancer_features = load_model(BREAST_CANCER_FEATURES_PATH)
lung_cancer_features = load_model(LUNG_CANCER_FEATURES_PATH)

@app.route('/')
def home():
    return jsonify({"message": "Cancer Prediction API is running"})

@app.route('/api/predict/breast', methods=['POST'])
def predict_breast_cancer():
    try:
        data = request.json
        logger.info(f"Received breast cancer prediction request: {data}")
        
        if breast_cancer_features is None:
            feature_names = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
            ]
            logger.warning(f"Using fallback feature list: {feature_names}")
        else:
            feature_names = breast_cancer_features
            logger.info(f"Using loaded feature list: {feature_names}")
        
        features = []
        for feature in feature_names:
            api_param = feature.strip().lower().replace(' ', '_')
            value = float(data.get(api_param, 0))
            features.append(value)
            logger.info(f"Feature {feature} ({api_param}): {value}")
        
        logger.info(f"Extracted features: {features}")
        
        if not all(isinstance(x, (int, float)) for x in features):
            return jsonify({"error": "All features must be numeric"}), 400
        
        if breast_cancer_scaler:
            try:
                features = breast_cancer_scaler.transform([features])
            except Exception as e:
                logger.error(f"Error during breast cancer scaling: {e}")
                return jsonify({"error": "Error during feature scaling. The model may need to be retrained."}), 500
        else:
            features = [features] 
        
        if breast_cancer_model:
            prediction = breast_cancer_model.predict(features)[0]
            probability = breast_cancer_model.predict_proba(features)[0]
            
            result = {
                "prediction": "Malignant" if prediction == 1 else "Benign",
                "probability": float(probability[1]) if prediction == 1 else float(probability[0]),
                "status": "success"
            }
            logger.info(f"Breast cancer prediction result: {result}")
            return jsonify(result)
        else:
            return jsonify({"error": "Model not available"}), 500
            
    except Exception as e:
        logger.error(f"Error in breast cancer prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/lung', methods=['POST'])
def predict_lung_cancer():
    try:
        data = request.json
        logger.info(f"Received lung cancer prediction request: {data}")
        
        if lung_cancer_features is None:
            feature_names = [
                'SMOKING', 'AGE', 'ALCOHOL CONSUMING', 'SHORTNESS OF BREATH',
                'CHEST PAIN', 'COUGHING', 'CHRONIC DISEASE'
            ]
            logger.warning(f"Using fallback feature list: {feature_names}")
        else:
            feature_names = lung_cancer_features
            logger.info(f"Using loaded feature list: {feature_names}")
        
        features = []
        for feature in feature_names:
            api_param = feature.lower().replace(' ', '_')
            api_param_stripped = feature.strip().lower().replace(' ', '_')
            api_param_no_spaces = feature.lower().replace(' ', '')
            
            if api_param in data:
                value = float(data.get(api_param, 0))
            elif api_param_stripped in data:
                value = float(data.get(api_param_stripped, 0))
            elif api_param_no_spaces in data:
                value = float(data.get(api_param_no_spaces, 0))
            else:
                found = False
                for key in data.keys():
                    if (api_param in key) or (api_param_stripped in key) or (api_param_no_spaces in key):
                        value = float(data.get(key, 0))
                        found = True
                        logger.info(f"Feature {feature} found using partial match with {key}")
                        break
                
                if not found:
                    value = 0.0
                    logger.warning(f"Feature {feature} not found in request data using any matching method")
            
            features.append(value)
            logger.info(f"Feature {feature}: {value}")
        
        logger.info(f"Extracted lung cancer features: {features}")
        
        if not all(isinstance(x, (int, float)) for x in features):
            return jsonify({"error": "All features must be numeric"}), 400
        
        if lung_cancer_scaler:
            try:
                features = lung_cancer_scaler.transform([features])
            except Exception as e:
                logger.error(f"Error during lung cancer scaling: {e}")
                return jsonify({"error": "Error during feature scaling. The model may need to be retrained."}), 500
        else:
            features = [features]  
        if lung_cancer_model:
            prediction = lung_cancer_model.predict(features)[0]
            probability = lung_cancer_model.predict_proba(features)[0]
            
            result = {
                "prediction": "Cancer Detected" if prediction == 1 else "No Cancer",
                "probability": float(probability[1]) if prediction == 1 else float(probability[0]),
                "status": "success"
            }
            logger.info(f"Lung cancer prediction result: {result}")
            return jsonify(result)
        else:
            return jsonify({"error": "Model not available"}), 500
            
    except Exception as e:
        logger.error(f"Error in lung cancer prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    health = {
        "status": "healthy",
        "models": {
            "breast_cancer": "loaded" if breast_cancer_model else "not loaded",
            "lung_cancer": "loaded" if lung_cancer_model else "not loaded"
        },
        "features": {
            "breast_cancer": breast_cancer_features if breast_cancer_features else "not loaded",
            "lung_cancer": lung_cancer_features if lung_cancer_features else "not loaded"
        },
        "scalers": {
            "breast_cancer": "fitted" if breast_cancer_scaler and check_scaler_is_fitted(breast_cancer_scaler, "Breast cancer") else "not fitted",
            "lung_cancer": "fitted" if lung_cancer_scaler and check_scaler_is_fitted(lung_cancer_scaler, "Lung cancer") else "not fitted"
        }
    }
    return jsonify(health)

@app.route('/api/info/<cancer_type>', methods=['GET'])
def cancer_info(cancer_type):
    if cancer_type == 'breast':
        if breast_cancer_features:
            return jsonify({
                "model": "breast_cancer",
                "features": breast_cancer_features,
                "status": "success"
            })
        else:
            return jsonify({
                "model": "breast_cancer",
                "features": "not available",
                "status": "warning"
            })
    elif cancer_type == 'lung':
        if lung_cancer_features:
            cleaned_features = [feature.strip() for feature in lung_cancer_features]
            return jsonify({
                "model": "lung_cancer",
                "features": lung_cancer_features,
                "cleaned_features": cleaned_features, 
                "api_parameters": [feature.strip().lower().replace(' ', '_') for feature in lung_cancer_features],
                "status": "success"
            })
        else:
            return jsonify({
                "model": "lung_cancer",
                "features": "not available",
                "status": "warning"
            })
    else:
        return jsonify({"error": "Invalid cancer type"}), 400

@app.route('/api/report/generate', methods=['POST'])
def generate_report():
    try:
        data = request.json
        logger.info(f"Generating PDF report for prediction: {data}")
        
        buffer = io.BytesIO()
        
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(72, height - 72, "Medical Prediction Report")
        
        pdf.setStrokeColorRGB(0.29, 0.61, 0.88)  # #4A90E2
        pdf.setLineWidth(2)
        pdf.line(72, height - 85, width - 72, height - 85)
        
        from datetime import datetime
        import uuid
        current_date = datetime.now()
        report_id = str(uuid.uuid4())[:8].upper()
        
        pdf.setFont("Helvetica", 11)
        pdf.drawString(72, height - 110, f"Report Date: {current_date.strftime('%B %d, %Y')}")
        pdf.drawString(72, height - 125, f"Report ID: {report_id}")
        pdf.drawString(72, height - 140, f"Time: {current_date.strftime('%H:%M:%S')}")
        
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(72, height - 180, "Prediction Details")
        
        pdf.setFont("Helvetica", 11)
        
        cancer_type = data.get('type', 'Unknown')
        pdf.drawString(72, height - 205, f"Cancer Type: {cancer_type.capitalize()} Cancer")
        
        prediction = data.get('prediction', 'Unknown')
        pdf.drawString(72, height - 220, f"Result: {prediction}")
        
        if prediction.lower() in ['benign', 'no cancer']:
            pdf.setFillColorRGB(0.2, 0.8, 0.2)  # Green
        else:
            pdf.setFillColorRGB(0.8, 0.2, 0.2)  # Red
        pdf.circle(340, height - 220, 6, fill=1)
        pdf.setFillColorRGB(0, 0, 0) 
        
        probability = data.get('probability', 0)
        pdf.drawString(72, height - 235, f"Probability: {(probability * 100):.2f}%")
        
        pdf.setLineWidth(1)
        pdf.rect(72, height - 255, 200, 10, stroke=1, fill=0)
        bar_width = int(200 * probability)
        
        if probability < 0.3:
            pdf.setFillColorRGB(0.2, 0.8, 0.2)  # Green
        elif probability < 0.7:
            pdf.setFillColorRGB(1.0, 0.7, 0.0)  # Amber
        else:
            pdf.setFillColorRGB(0.8, 0.2, 0.2)  # Red
            
        pdf.rect(72, height - 255, bar_width, 10, stroke=0, fill=1)
        pdf.setFillColorRGB(0, 0, 0)  # Reset to black
        
        pdf.setStrokeColorRGB(0.8, 0.8, 0.8)  # Light grey
        pdf.setLineWidth(1)
        pdf.line(72, height - 275, width - 72, height - 275)
        
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(72, height - 295, "Medical Recommendations")
        
        pdf.setFont("Helvetica", 11)
        recommendations = []
        if prediction.lower() in ['benign', 'no cancer']:
            recommendations = [
                "Continue with regular medical check-ups as recommended for your age group.",
                "Maintain a healthy lifestyle with balanced nutrition and regular exercise.",
                "Monitor for any changes in symptoms and report them to your healthcare provider.",
                "Follow the standard cancer screening guidelines appropriate for your age and risk factors."
            ]
        else:
            recommendations = [
                "Consult with a healthcare professional immediately to discuss these results.",
                "Further diagnostic tests may be needed to confirm this prediction.",
                "Develop a treatment plan with your healthcare provider based on additional testing.",
                "Consider seeking a second opinion from another specialist.",
                "Look into support groups and counseling resources for emotional support."
            ]
        
        y_position = height - 320
        for recommendation in recommendations:
            pdf.drawString(90, y_position, "• " + recommendation)
            y_position -= 20
        
        y_position -= 30
        pdf.setStrokeColorRGB(0.8, 0.8, 0.8)
        pdf.rect(72, y_position - 60, width - 144, 60, stroke=1, fill=0)
        
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(82, y_position - 15, "Disclaimer:")
        pdf.setFont("Helvetica", 9)
        pdf.drawString(82, y_position - 30, "This prediction report is generated by an AI model and is intended for informational purposes only.")
        pdf.drawString(82, y_position - 45, "It should not be considered as a medical diagnosis. Please consult with qualified healthcare")
        pdf.drawString(82, y_position - 60, "professionals for proper diagnosis, advice, and treatment.")
        
        pdf.setFont("Helvetica", 9)
        pdf.drawString(width/2 - 40, 40, f"Page 1 of 1 | Report ID: {report_id}")
        
        pdf.setStrokeColorRGB(0.29, 0.61, 0.88) 
        pdf.setLineWidth(2)
        pdf.line(72, 30, width - 72, 30)
        pdf.save()
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=cancer_prediction_report_{report_id}.pdf'
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return jsonify({"error": str(e)}), 500
    
def generate_test_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import os
    import numpy as np
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    breast_top_features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 
        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ]
    lung_top_features = ['AGE', 'ALCOHOL CONSUMING', 'ALLERGY', 'PEER_PRESSURE', 'YELLOW_FINGERS', 'FATIGUE', 'COUGHING']
    
    breast_model = RandomForestClassifier(n_estimators=100, random_state=42)
    breast_scaler = StandardScaler()
    
    breast_dummy_data = np.random.rand(100, len(breast_top_features))
    breast_dummy_labels = np.random.choice([0, 1], size=100)
    
    breast_scaler.fit(breast_dummy_data)
    breast_model.fit(breast_scaler.transform(breast_dummy_data), breast_dummy_labels)
    
    lung_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lung_scaler = StandardScaler()
    
    lung_dummy_data = np.random.rand(100, len(lung_top_features))
    lung_dummy_labels = np.random.choice([0, 1], size=100)
    
    lung_scaler.fit(lung_dummy_data)
    lung_model.fit(lung_scaler.transform(lung_dummy_data), lung_dummy_labels)
    with open(BREAST_CANCER_MODEL_PATH, 'wb') as f:
        pickle.dump(breast_model, f)
    
    with open(LUNG_CANCER_MODEL_PATH, 'wb') as f:
        pickle.dump(lung_model, f)
    
    with open(BREAST_CANCER_SCALER_PATH, 'wb') as f:
        pickle.dump(breast_scaler, f)
    
    with open(LUNG_CANCER_SCALER_PATH, 'wb') as f:
        pickle.dump(lung_scaler, f)
    
    with open(BREAST_CANCER_FEATURES_PATH, 'wb') as f:
        pickle.dump(breast_top_features, f)
    
    with open(LUNG_CANCER_FEATURES_PATH, 'wb') as f:
        pickle.dump(lung_top_features, f)
    
    print("Test models generated successfully")

if __name__ == "__main__":
    should_regenerate_models = False
    
    if not (os.path.exists(BREAST_CANCER_MODEL_PATH) and os.path.exists(LUNG_CANCER_MODEL_PATH)):
        logger.info("Models not found. Generating test models.")
        should_regenerate_models = True
    
    if breast_cancer_scaler and not check_scaler_is_fitted(breast_cancer_scaler, "Breast cancer"):
        logger.warning("Breast cancer scaler not fitted properly.")
        should_regenerate_models = True
    
    if lung_cancer_scaler and not check_scaler_is_fitted(lung_cancer_scaler, "Lung cancer"):
        logger.warning("Lung cancer scaler not fitted properly.")
        should_regenerate_models = True
    
    if should_regenerate_models:
        generate_test_models()
        breast_cancer_model = load_model(BREAST_CANCER_MODEL_PATH)
        lung_cancer_model = load_model(LUNG_CANCER_MODEL_PATH)
        breast_cancer_scaler = load_model(BREAST_CANCER_SCALER_PATH)
        lung_cancer_scaler = load_model(LUNG_CANCER_SCALER_PATH)
        breast_cancer_features = load_model(BREAST_CANCER_FEATURES_PATH)
        lung_cancer_features = load_model(LUNG_CANCER_FEATURES_PATH)
    
    if breast_cancer_features:
        logger.info(f"Loaded breast cancer features: {breast_cancer_features}")
    else:
        logger.warning("Warning: Breast cancer features not loaded")
        
    if lung_cancer_features:
        logger.info(f"Loaded lung cancer features: {lung_cancer_features}")
        cleaned_features = [feature.strip().lower().replace(' ', '_') for feature in lung_cancer_features]
        logger.info(f"API parameter names for lung cancer: {cleaned_features}")
    else:
        logger.warning("Warning: Lung cancer features not loaded")
    
    app.run(debug=True, host='0.0.0.0', port=5000)