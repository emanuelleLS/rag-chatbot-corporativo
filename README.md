# RAG Chatbot Corporativo 📄🤖

Este projeto implementa um **chatbot corporativo baseado em RAG (Retrieval-Augmented Generation)**, capaz de responder perguntas de colaboradores **exclusivamente com base em documentos internos da empresa**, garantindo confiabilidade, rastreabilidade e controle de contexto.

O sistema simula o ambiente de uma empresa de grande porte, com múltiplos departamentos (RH, TI, etc.) e políticas internas organizadas em documentos PDF.

---

## 🎯 Objetivo

Permitir que colaboradores tirem dúvidas internas de forma rápida e segura, como por exemplo:

- Como devo solicitar minhas férias?
- Quais são as regras de trabalho remoto?
- Como configuro a VPN corporativa?

O chatbot **não inventa respostas**. Caso a informação não esteja presente nos documentos carregados, ele informa explicitamente que não encontrou a resposta.

---

## 🧠 Arquitetura da Solução

A solução segue o padrão moderno de RAG utilizado em ambientes corporativos:

- Documentos PDF organizados por departamento
- Divisão dos documentos em chunks semânticos
- Geração de embeddings locais
- Armazenamento vetorial com ChromaDB
- Busca semântica com filtros por metadados
- Uso de LLM apenas para geração final da resposta
- Retorno das fontes utilizadas na resposta

Fluxo resumido:

PDFs → Chunks → Embeddings → Vector Store → Busca Semântica → LLM → Resposta + Fontes

---

## 🧱 Tecnologias Utilizadas

- Python 3.11+
- LangChain
- ChromaDB
- Sentence Transformers
- Google Gemini
- PyPDF
- VS Code

---

## 📁 Estrutura do Projeto

rag-chatbot-corporativo/
│
├── main.py
├── documentos_empresa/
│   ├── RH/
│   │   ├── politica_ferias.pdf
│   │   └── politica_home_office.pdf
│   └── TI/
│       ├── procedimento_vpn.pdf
│       └── politica_seguranca_informacao.pdf
│
├── .gitignore
├── .env.example
└── README.md

---

## ⚙️ Como Executar

### 1. Clonar o repositório

git clone https://github.com/emanuelleLS/rag-chatbot-corporativo.git
cd rag-chatbot-corporativo

### 2. Criar e ativar o ambiente virtual

python -m venv .venv
.venv\Scripts\activate

### 3. Instalar as dependências

pip install -r requirements.txt

### 4. Configurar variáveis de ambiente

Crie um arquivo .env com base no .env.example e informe sua chave da API do Google Gemini.

### 5. Executar o projeto

python main.py

---

## 🔐 Confiabilidade da Informação

- Respostas geradas apenas com base nos documentos
- Filtros por departamento
- Indicação explícita das fontes
- Proteção contra alucinações

---

## 🚀 Próximos Passos

- API REST com FastAPI
- Interface web
- Upload dinâmico de documentos
- Controle de acesso por departamento

---

## 👩‍💻 Autora

Emanuelle Scheifer  
Engenharia de Software | IA Aplicada | Sistemas RAG
