#Importando librerias
import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.7, max_tokens=250)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil. Responde a la pregunta basándote ÚNICAMENTE en el texto proporcionado a continuación. Si la respuesta no se encuentra en el texto, responde 'No tengo información sobre eso'.\n\nTEXTO:\n{contexto}"),
    ("human", "{pregunta}")
])

# Unimos el prompt y el modelo en una cadena (chain)
cadena = prompt | model

def consultar_noticia(url, pregunta):
    """Función que llama Gradio cuando el usuario presione 'Submit'"""
    try:
        # cargando el contenido de la url seleccionada
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # limpiando el texto
        texto_sucio = documents[0].page_content
        contexto = re.sub(r'\s+', ' ', texto_sucio).strip()
        

        response = cadena.invoke({"contexto": contexto, "pregunta": pregunta})
        return response.content
        
    except Exception as e:
        return f"Ocurrió un error: {str(e)}"

# Interfaz visual
demo = gr.Interface(
    fn=consultar_noticia,
    inputs=[
        gr.Textbox(label="URL de la noticia", lines=1, placeholder="Ingresa el link aquí..."),
        gr.Textbox(label="Pregunta", lines=2, placeholder="Ej: Resume en 3 puntos, o ¿De qué trata?")
    ],
    outputs=[gr.Textbox(label="Respuesta de la IA", lines=8)],
    flagging_mode="never",
    title="📰 Asistente para resumir noticias",
    description="Introduce la URL de una noticia y pregúntale lo que quieras basándote en su contenido."
)

if __name__ == "__main__":
    demo.launch()
