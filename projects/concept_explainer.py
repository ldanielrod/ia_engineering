import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import gradio as gr


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


prompt_template_str = """
Tu tarea es explicarme el concepto de **{concept}** de una manera que sea:

1. Clara e intuitiva
2. Concisa (en menos de 100 palabras)
3. Adaptada específicamente para mí y lo que ya sé

Usa la siguiente información sobre mí para personalizar tus explicaciones:

- Rol: Data & Solutions Architect en la industria de energetica
- Intereses profesionales: Construcción de asistentes conversacionales
- Contexto técnico: Arquitectura de datos y soluciones en el sector energético

La personalización debe ser sutil y natural. Evita referencias forzadas a mi background o intereses que no mejoren genuinamente mi comprensión del concepto.
"""
# Crear un prompt template
prompt_template = PromptTemplate.from_template(prompt_template_str)

# Crear una interfaz del modelo
model = init_chat_model("gpt-4o-mini", model_provider="openai")


def generate_explanation(input_text):
  prompt = prompt_template.format(concept=input_text)
  response = model.invoke(prompt)
  return response.content



demo = gr.Interface(
    fn=generate_explanation,
    inputs=[gr.Textbox(label="Concepto a explicar", lines=1)],
    outputs=[gr.Textbox(label="Explicación", lines=5)],
    flagging_mode="never",
    title="Explicador de Conceptos con IA",
    description="Introduce un término técnico y obtén una explicación personalizada"
)

demo.launch()
