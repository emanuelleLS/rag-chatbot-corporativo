<template>
    <div class="chat-container">
        <h1>RAG Chatbot Corporativo</h1>
        <div class="messages">
            <div v-for="(msg, index) in mensagens" :key="index" :class="msg.tipo">
                <strong>{{ msg.tipo === 'usuario' ? 'Você' : 'Chatbot' }}:</strong>
                <p>{{ msg.texto }}</p>
                <ul v-if="msg.fontes && msg.fontes.length">
                    <li v-for="(f, i) in msg.fontes" :key="i">{{ f }}</li>
                </ul>
            </div>
        </div>

        <form @submit.prevent="enviarPergunta">
            <input v-model="pergunta" placeholder="Digite sua pergunta..." />
            <button type="submit">Enviar</button>
        </form>
    </div>
</template>

<script>
import axios from "axios";

export default {
    data() {
        return {
            pergunta: "",
            mensagens: [],
        };
    },
    methods: {
        async enviarPergunta() {
            if (!this.pergunta.trim()) return;

            this.mensagens.push({ tipo: "usuario", texto: this.pergunta });
            const perguntaAtual = this.pergunta;
            this.pergunta = "";

            try {
                const response = await axios.post("http://127.0.0.1:8000/perguntar", {
                    pergunta: perguntaAtual,
                });

                this.mensagens.push({
                    tipo: "chatbot",
                    texto: response.data.resposta,
                    fontes: response.data.fontes,
                });
            } catch (error) {
                this.mensagens.push({
                    tipo: "chatbot",
                    texto: "Erro ao se comunicar com o servidor.",
                    fontes: [],
                });
            }
        },
    },
};
</script>

<style scoped>
.chat-container {
    max-width: 600px;
    margin: 0 auto;
    font-family: Arial, sans-serif;
}

.messages {
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 10px;
}

.usuario {
    text-align: right;
    margin-bottom: 10px;
}

.chatbot {
    text-align: left;
    margin-bottom: 10px;
}

input {
    width: 80%;
    padding: 8px;
}

button {
    padding: 8px 12px;
}
</style>
