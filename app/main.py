from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
# ... (mantenha seus outros imports como joblib, pandas, etc.)

# 1. Configuração de metadados da API para melhorar o UX inicial
app = FastAPI(
    title="🛡️ Sistema de Análise de Risco de Crédito",
    description="""
    API corporativa para avaliação de propostas de crédito utilizando um modelo preditivo baseado em **XGBoost**.
    
    ### Como testar:
    1. Abra o endpoint **POST `/predict`** abaixo.
    2. Clique em **Try it out**.
    3. Cole um dos cenários de teste documentados no repositório.
    4. Avalie o retorno de risco (`LOW_RISK`, `HIGH_RISK`).
    """,
    version="1.0.0",
    docs_url=None, # Desativamos a rota padrão para customizar o tema abaixo
    redoc_url=None
)

# Definindo tags para organizar os blocos visuais
tags_metadata = [
    {
        "name": "Análise Preditiva",
        "description": "Endpoints responsáveis por interagir com o modelo XGBoost e inferir o risco.",
    },
    {
        "name": "Monitoramento",
        "description": "Verificação de integridade e status do serviço (Healthcheck).",
    }
]
app.openapi_tags = tags_metadata

# 2. Injetando uma folha de estilo customizada (Swagger Themes)
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        # Usando um tema dark/moderno do Swagger disponível publicamente por CDN
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-themes@3.0.0/themes/3.x/theme-flattop.css",
    )

# 3. Organizando os seus Endpoints atuais com as novas Tags e descrições
@app.get("/", tags=["Monitoramento"], summary="Verifica se a API está online")
def home():
    return {"status": "API Online", "version": "1.0.0"}

@app.post("/predict", tags=["Análise Preditiva"], summary="Executa a classificação de risco do cliente")
def predict_credit(request: CreditRequest):
    # ... (mantenha a lógica do seu modelo aqui)
    pass