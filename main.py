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
        "RH": [
            "férias", "ferias", "benefício", "beneficios", "salário", "salario",
            "home office", "remoto", "folga", "licença", "licenca", "ponto"
        ],
        "TI": [
            "vpn", "senha", "acesso", "login", "sistema", "email",
            "computador", "rede", "segurança", "seguranca"
        ]
    }

    for departamento, palavras in regras.items():
        if any(p in texto for p in palavras):
            return departamento

    return None



def responder_pergunta(pergunta, vector_db, llm):
    departamento = detectar_departamento(pergunta)
    filtro = {"departamento": departamento} if departamento else None

    docs_scores = vector_db.similarity_search_with_score(
        pergunta,
        k=3,
        filter=filtro
    )

    docs_relevantes = [doc for doc, score in docs_scores if score <= 1.5]
    
    if not docs_relevantes:
        docs_relevantes = [doc for doc, _ in docs_scores[:1]]


    if not docs_relevantes:
        return "Não encontrei essa informação nos documentos disponíveis.", []

    contexto, fontes = montar_contexto(docs_relevantes)

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
            bloco.get("text", "") for bloco in conteudo if bloco.get("type") == "text"
        )

    return conteudo.strip(), fontes




documentos_empresa = []

documentos_empresa.extend(
    carregar_pdf("documentos_empresa/RH/politica_ferias.pdf", "RH", "politica", "1.0")
)

documentos_empresa.extend(
    carregar_pdf("documentos_empresa/RH/politica_home_office.pdf", "RH", "politica", "1.0")
)

documentos_empresa.extend(
    carregar_pdf("documentos_empresa/TI/procedimento_vpn.pdf", "TI", "procedimento", "1.0")
)

documentos_empresa.extend(
    carregar_pdf("documentos_empresa/TI/politica_seguranca_informacao.pdf", "TI", "politica", "1.0")
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

documentos_chunked = text_splitter.split_documents(documentos_empresa)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    documentos_chunked,
    embedding=embeddings
)


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0
)


pergunta = "Como configuro a VPN?"

resposta, fontes = responder_pergunta(pergunta, vector_db, llm)

print("Pergunta:")
print(pergunta)

print("\nResposta do chatbot:")
print(resposta)

print("\nFontes:")
for f in fontes:
    print("-", f)
