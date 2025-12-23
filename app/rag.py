from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


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
        "RH": ["férias", "ferias", "benefício", "salário", "home office", "folga", "licença", "ponto"],
        "TI": ["vpn", "senha", "acesso", "login", "sistema", "email", "computador", "rede", "segurança"]
    }

    for departamento, palavras in regras.items():
        if any(p in texto for p in palavras):
            return departamento

    return None


def inicializar_rag():
    documentos = []

    documentos.extend(carregar_pdf("documentos_empresa/RH/politica_ferias.pdf", "RH", "politica", "1.0"))
    documentos.extend(carregar_pdf("documentos_empresa/RH/politica_home_office.pdf", "RH", "politica", "1.0"))
    documentos.extend(carregar_pdf("documentos_empresa/TI/procedimento_vpn.pdf", "TI", "procedimento", "1.0"))
    documentos.extend(carregar_pdf("documentos_empresa/TI/politica_seguranca_informacao.pdf", "TI", "politica", "1.0"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embedding=embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    return vector_db, llm


def responder_pergunta(pergunta, vector_db, llm):
    departamento = detectar_departamento(pergunta)
    filtro = {"departamento": departamento} if departamento else None

    docs_scores = vector_db.similarity_search_with_score(pergunta, k=3, filter=filtro)
    docs = [doc for doc, score in docs_scores if score <= 1.5] or [docs_scores[0][0]]

    contexto, fontes = montar_contexto(docs)

    prompt = f"""
Você é um assistente corporativo interno.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta objetiva e profissional:
"""

    resposta = llm.invoke(prompt)
    conteudo = resposta.content

    if isinstance(conteudo, list):
        conteudo = "".join(b.get("text", "") for b in conteudo if b.get("type") == "text")

    return conteudo.strip(), fontes
