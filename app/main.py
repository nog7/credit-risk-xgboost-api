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
    # 1. Converte o request JSON para o DataFrame no formato esperado pelo pipeline
    input_data = pd.DataFrame([request.dict()])
    
    # 2. Captura apenas a probabilidade de inadimplência (classe 1)
    # predict_proba retorna uma matriz: [prob_classe_0, prob_classe_1]
    prob_inadimplencia = model.predict_proba(input_data)[0][1]
    
    # 3. Aplica a Regra de Negócio Pessoal / Threshold Dinâmico
    if prob_inadimplencia <= 0.45:
        approved = True
        risk_status = "LOW_RISK"
        recommendation = "AUTOMATIC_APPROVAL"
        
    elif 0.45 < prob_inadimplencia <= 0.65:
        approved = False  # Segura a aprovação automática por segurança
        risk_status = "MODERATE_RISK"
        recommendation = "MANUAL_REVISION_REQUIRED"  # Vai para a mesa de análise
        
    else:
        approved = False
        risk_status = "HIGH_RISK"
        recommendation = "AUTOMATIC_REFUSAL"
        
    # 4. Retorna o novo contrato da API muito mais maduro
    return {
        "approved": approved,
        "default_probability": round(float(prob_inadimplencia), 4),
        "risk_grade": risk_status,
        "action_required": recommendation
    }