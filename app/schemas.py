from pydantic import BaseModel


class PerguntaRequest(BaseModel):
    pergunta: str


class RespostaResponse(BaseModel):
    resposta: str
    fontes: list[str]
