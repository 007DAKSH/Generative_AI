# app.py
import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load your trained model
model_path = "./trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_text(prompt, max_length=100, temperature=0.7, top_p=0.9):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a prompt here..."),
        gr.Slider(20, 200, value=100, step=10, label="Max Length"),
        gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p (nucleus sampling)"),
    ],
    outputs="text",
    title="📝 Text Generator",
    description="Fine-tuned GPT-2 model running locally with Gradio!"
)

if __name__ == "__main__":
    demo.launch()
