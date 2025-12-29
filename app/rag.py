import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

PERSIST_DIR = "vectorstore"


def carregar_pdf(caminho, departamento, tipo, versao):
    loader = PyPDFLoader(caminho)
    documentos = loader.load()
    for doc in documentos:
        doc.metadata["departamento"] = departamento
        doc.metadata["tipo"] = tipo
        doc.metadata["versao"] = versao
        doc.metadata["fonte"] = caminho.split("/")[-1]
    return documentos


def montar_contexto(documentos):
    contexto = ""
    fontes = set()
    for doc in documentos:
        contexto += doc.page_content + "\n\n"
        fontes.add(
            f"{doc.metadata['departamento']} - {doc.metadata['fonte']} - v{doc.metadata['versao']} - pág. {doc.metadata.get('page', 0) + 1}"
        )
    return contexto, list(fontes)


def detectar_departamento(pergunta):
    texto = pergunta.lower()
    regras = {
        "RH": [
            "férias",
            "ferias",
            "benefício",
            "beneficios",
            "salário",
            "salario",
            "home office",
            "remoto",
            "trabalho remoto",
            "teletrabalho",
            "folga",
            "licença",
            "licenca",
            "ponto",
        ],
        "TI": [
            "vpn",
            "senha",
            "acesso",
            "login",
            "sistema",
            "email",
            "computador",
            "rede",
            "segurança",
            "seguranca",
        ],
    }

    for departamento, palavras in regras.items():
        if any(p in texto for p in palavras):
            return departamento
    return None


def inicializar_rag():
    if Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)

    documentos = carregar_pdf(
        "documentos_empresa/RH/RH_politica_unificada_NovaDrive_Technologies_v1.0.pdf",
        "RH",
        "politica",
        "1.0",
    )

    documentos.extend(
        carregar_pdf(
            "documentos_empresa/TI/procedimento_vpn.pdf", "TI", "procedimento", "1.0"
        )
    )
    documentos.extend(
        carregar_pdf(
            "documentos_empresa/TI/politica_seguranca_informacao.pdf",
            "TI",
            "politica",
            "1.0",
        )
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)

    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, timeout=30)
    return vector_db, llm



def selecionar_chunks_relevantes(docs_scores, top_docs=3):
    """
    Seleciona apenas o chunk mais relevante por documento (fonte),
    evitando que múltiplos chunks do mesmo PDF sejam incluídos.
    """
    docs_por_fonte = {}
    for doc, score in docs_scores:
        fonte = doc.metadata["fonte"]
        if fonte not in docs_por_fonte or score < docs_por_fonte[fonte][1]:
            docs_por_fonte[fonte] = (doc, score)

    docs_ordenados = sorted(docs_por_fonte.values(), key=lambda x: x[1])
    return [doc for doc, _ in docs_ordenados[:top_docs]]


def responder_pergunta(pergunta, vector_db, llm):
    departamento = detectar_departamento(pergunta)
    filtro = {"departamento": departamento} if departamento else None

    docs_scores = vector_db.similarity_search_with_score(pergunta, k=10, filter=filtro)
    if not docs_scores:
        return "Não encontrei essa informação nos documentos disponíveis.", []

    docs = selecionar_chunks_relevantes(docs_scores, top_docs=3)
    contexto, fontes = montar_contexto(docs)

    prompt = f"""
Você é um assistente corporativo interno.

Responda à pergunta abaixo somente com base nas informações fornecidas no contexto.
Se a resposta não estiver claramente no contexto, diga:
"Não encontrei essa informação nos documentos disponíveis."

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta objetiva e profissional:
"""

    resposta = llm.invoke(prompt)
    conteudo = resposta.content

    if isinstance(conteudo, list):
        conteudo = "".join(
            b.get("text", "") for b in conteudo if b.get("type") == "text"
        )

    conteudo = conteudo.strip()
    if not conteudo:
        return "Não encontrei essa informação nos documentos disponíveis.", fontes

    return conteudo, fontes
