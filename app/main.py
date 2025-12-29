from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.rag import inicializar_rag, responder_pergunta
from app.schemas import PerguntaRequest, RespostaResponse

app = FastAPI(title="RAG Chatbot Corporativo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_db, llm = inicializar_rag()


@app.post("/perguntar", response_model=RespostaResponse)
def perguntar(payload: PerguntaRequest):
    resposta, fontes = responder_pergunta(payload.pergunta, vector_db, llm)
    return {"resposta": resposta, "fontes": fontes}
