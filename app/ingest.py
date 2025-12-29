from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def carregar_pdf(caminho, departamento, tipo, versao):
    loader = PyPDFLoader(caminho)
    documentos = loader.load()
    for doc in documentos:
        doc.metadata["departamento"] = departamento
        doc.metadata["tipo"] = tipo
        doc.metadata["versao"] = versao
        doc.metadata["fonte"] = caminho.split("/")[-1]
    return documentos


def montar_vectorstore(documentos, chunk_size=500, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma.from_documents(chunks, embedding=embeddings)

    return vector_db


if __name__ == "__main__":
    documentos = []

    documentos.extend(
        carregar_pdf(
            "documentos_empresa/RH/politica_ferias.pdf", "RH", "politica", "1.0"
        )
    )
    documentos.extend(
        carregar_pdf(
            "documentos_empresa/RH/politica_home_office.pdf", "RH", "politica", "1.0"
        )
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

    vector_db = montar_vectorstore(documentos)

    vector_db.persist("db_chroma")
    print("Vector store criada e salva em 'db_chroma'.")
