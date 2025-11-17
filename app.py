import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel


BASE_MODEL = "ibm-granite/granite-4.0-micro"  #  base model
ADAPTER_REPO = "owaiskha9654/granite-4-finetuned-FairyTaleQA"  # Fine Tuned LoRA repo
# -----------------------------

# Download adapter folder locally
from huggingface_hub import snapshot_download
adapter_path = snapshot_download(repo_id=ADAPTER_REPO)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
    offload_folder="/tmp/offload",      # writes some tensors to disk
    offload_state_dict=True,
)

# Load LoRA adapters
peft_model = PeftModel.from_pretrained(
    base,
    adapter_path,
    device_map="auto",
    offload_folder="/tmp/offload",  
    offload_state_dict=True,
)

peft_model.eval()

# Pipeline
pipe = pipeline(
    "text-generation",
    model=peft_model,
    tokenizer=tokenizer,
)

def generate(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
    result = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    return result[0]["generated_text"]


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=4, label="Prompt"),
        gr.Slider(10, 1024, value=200, label="Max New Tokens"),
        gr.Slider(0.1, 1.5, value=0.7, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, label="Top-p"),
    ],
    outputs="text",
    title="Granite-4 LoRA Fine-Tuned Model",
)

demo.launch(share = True)
