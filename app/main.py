import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Definição do Schema de Entrada (Pydantic)
class CreditRequest(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# 2. Inicialização do FastAPI (Versão Padrão Sem Firulas)
app = FastAPI(
    title="API de Análise de Risco de Crédito",
    version="1.0.0"
)

# Mantemos o CORS ativo para o caso de você querer conectar qualquer coisa no futuro
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Carregamento do Modelo e Scaler (.joblib com plano de contingência)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_model.joblib")

model = None
scaler = None
features = None

try:
    loaded_data = joblib.load(MODEL_PATH)
    
    if isinstance(loaded_data, dict):
        model = loaded_data.get("model")
        scaler = loaded_data.get("scaler")
        features = loaded_data.get("features")
    else:
        model = loaded_data
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# 4. Endpoints
@app.get("/")
def home():
    return {"status": "API Online", "version": "1.0.0"}

@app.post("/predict")
def predict_credit(request: CreditRequest):
    if model is None:
        return {"error": "Modelo preditivo não carregado no servidor."}
        
    data_dict = request.model_dump()
    df_input = pd.DataFrame([data_dict])
    
    if features is not None and scaler is not None:
        df_encoded = pd.get_dummies(df_input)
        for col in features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[features]
        X_final = scaler.transform(df_encoded)
    else:
        df_encoded = pd.get_dummies(df_input)
        X_final = df_encoded.values
    
    try:
        prediction = int(model.predict(X_final)[0])
        
        try:
            probability = float(model.predict_proba(X_final)[0][1])
        except:
            probability = 1.0 if prediction == 1 else 0.0
            
        risk_label = "HIGH_RISK" if prediction == 1 else "LOW_RISK"
        
        return {
            "prediction": prediction,
            "risk_status": risk_label,
            "inadimplencia_probability": round(probability, 4)
        }
    except Exception as e:
        return {"error": f"Erro na inferência: {str(e)}"}