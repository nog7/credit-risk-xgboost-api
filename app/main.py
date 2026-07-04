import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel

# 1. Definição do Schema de Entrada (Pydantic) para validação dos dados
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

# 2. Inicialização do FastAPI com metadados customizados para o UX
app = FastAPI(
    title="🛡️ Sistema de Análise de Risco de Crédito",
    description="""
    API corporativa para avaliação de propostas de crédito utilizando um modelo preditivo baseado em **XGBoost**.
    
    ### Como testar:
    1. Abra o endpoint **POST `/predict`** logo abaixo.
    2. Clique no botão **Try it out**.
    3. Altere os valores do JSON ou use um cenário de teste do repositório.
    4. Clique em **Execute** e avalie o retorno de risco (`LOW_RISK` ou `HIGH_RISK`).
    """,
    version="1.0.0",
    docs_url=None,       # Desativa a rota padrão para injetarmos o tema customizado
    redoc_url=None
)

# Configuração das tags visuais para organizar a interface
tags_metadata = [
    {
        "name": "Análise Preditiva",
        "description": "Endpoints responsáveis por interagir com o modelo XGBoost e inferir o risco.",
    },
    {
        "name": "Monitoramento",
        "description": "Verificação de integridade e status de operação do serviço (Healthcheck).",
    }
]
app.openapi_tags = tags_metadata

# 3. Carregamento Seguro do Modelo e Scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_model.pkl")

# Carrega o modelo de forma global para otimizar a performance da API
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data["scaler"]
    features = model_data["features"]
except Exception as e:
    print(f"Erro crítico ao carregar os artefatos do modelo: {e}")
    model, scaler, features = None, None, None

# 4. Rota Customizada para renderizar o Swagger UI com Tema Escuro (Moderno)
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-themes@3.0.0/themes/3.x/theme-flattop.css",
    )

# 5. Endpoints da Aplicação
@app.get("/", tags=["Monitoramento"], summary="Verifica se a API está online")
def home():
    return {"status": "API Online", "version": "1.0.0"}

@app.post("/predict", tags=["Análise Preditiva"], summary="Executa a classificação de risco do cliente")
def predict_credit(request: CreditRequest):
    if model is None or scaler is None:
        return {"error": "Modelo preditivo não inicializado corretamente no servidor."}
        
    # Converte a requisição recebida em um DataFrame do Pandas
    data_dict = request.model_dump()
    df_input = pd.DataFrame([data_dict])
    
    # Aplica o One-Hot Encoding exatamente como foi feito no treinamento do modelo
    df_encoded = pd.get_dummies(df_input)
    
    # Garante que a API possua exatamente as mesmas colunas estruturais que o modelo espera
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Reordena as colunas para bater com a ordem exata do treinamento
    df_encoded = df_encoded[features]
    
    # Aplica a padronização dos dados usando o Scaler treinado
    X_scaled = scaler.transform(df_encoded)
    
    # Executa a predição do XGBoost
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])
    
    # Define o rótulo de resposta baseado na saída binária do modelo
    risk_label = "HIGH_RISK" if prediction == 1 else "LOW_RISK"
    
    return {
        "prediction": prediction,
        "risk_status": risk_label,
        "inadimplencia_probability": round(probability, 4)
    }