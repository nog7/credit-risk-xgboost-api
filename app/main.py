from fastapi import FastAPI
import joblib
import pandas as pd
from app.schemas import CreditRequest

# 1. Inicializar o FastAPI
app = FastAPI(
    title="API de Análise de Risco de Crédito",
    description="API interna para validação de crédito usando XGBoost",
    version="1.0.0"
)

# 2. Carregar o pipeline completo de ML (Roda apenas UMA vez no startup da API)
MODEL_PATH = "models/credit_risk_model.joblib"
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"status": "API Online", "version": "1.0.0"}

@app.post("/predict")
def predict_credit(payload: CreditRequest):
    # 3. Converte o payload do Pydantic para dicionário e depois para DataFrame
    data = payload.model_dump()
    df_input = pd.DataFrame([data])
    
    # 4. Calcular a probabilidade de inadimplência (Classe 1)
    probabilities = model.predict_proba(df_input)[0]
    default_prob = float(probabilities[1])
    
    # 5. Regra de Negócio: Se o risco for menor que 40%, está aprovado
    threshold = 0.40
    approved = default_prob < threshold
    
    # 6. Definir faixas de risco comerciais para o retorno
    if default_prob < 0.15:
        risk_grade = "LOW_RISK"
    elif default_prob < 0.40:
        risk_grade = "MEDIUM_RISK"
    else:
        risk_grade = "HIGH_RISK"

    return {
        "approved": approved,
        "default_probability": round(default_prob, 4),
        "risk_grade": risk_grade
    }