import os
from huggingface_hub import snapshot_download
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ---------------- CONFIG ----------------
BASE_MODEL = "ibm-granite/granite-4.0-micro"
ADAPTER_REPO = "owaiskha9654/granite-4-finetuned-finance-qa-data"
HEADER_IMAGE = "https://github.com/Owaiskhan9654/Granite-4.0-Fine-Tuning/blob/669fcfa2c9e5c42d7ff67bac5ae341ce27a22fe9/ibm-granite-4-0-release.jpeg?raw=true"
OFFLOAD_DIR = "/tmp/offload"
# ----------------------------------------

os.makedirs(OFFLOAD_DIR, exist_ok=True)

# download adapter files
adapter_path = snapshot_download(repo_id=ADAPTER_REPO)

# load tokenizer (prefer adapter tokenizer if present)
try:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False, trust_remote_code=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)

# load base model with graceful fallbacks to avoid OOM in Spaces
load_failed = None
try:
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        # load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
except Exception as e4:
    load_failed = e4
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            # load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e8:
        load_failed = e8
        # final fallback: normal fp16 with disk offload
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype="auto",
            offload_folder=OFFLOAD_DIR,
            offload_state_dict=True,
            trust_remote_code=True,
        )

# attach LoRA adapter
peft_model = PeftModel.from_pretrained(base, adapter_path, device_map="auto")
peft_model.eval()

# create pipeline
pipe = pipeline("text-generation", model=peft_model, tokenizer=tokenizer, device_map="auto")

# generate function
def generate(prompt, max_new_tokens, temperature, top_p):
    prompt = (prompt or "").strip()
    if not prompt:
        return "Enter a prompt."
    out = pipe(
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=True,
        return_full_text=False,
    )
    return out[0]["generated_text"]

# --------- UI ----------
css = """
.gradio-container { max-width:1600px; margin:12px auto; }
.header-row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.header-title { font-size:32px; font-weight:700; color:#fff; }
.card { background:#0f1113; padding:16px; border-radius:12px; }
.output textarea { height: 320px !important; white-space: pre-wrap; }
#output_title { padding-left: 12px; }
"""

with gr.Blocks(title="Granite-4 LoRA", css=css, theme=gr.themes.Base()) as demo:
    with gr.Row(elem_classes="header-row"):
        gr.Markdown(f"### <span class='header-title'>Granite 4.0 LoRA on<br>Finance QA Benchmark Dataset</span>")
        gr.Image(value=HEADER_IMAGE,show_label=False,elem_id="header_img",interactive=False,width=450,height=300)
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                prompt = gr.Textbox(lines=15, label="Prompt", placeholder="### Instruction:\nSummarize the text.\n### Input:\n...")
                with gr.Row():
                    max_tokens = gr.Slider(10, 1024, value=100, label="Max New Tokens")
                    temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### Output", elem_id="output_title")
                output = gr.Textbox(value="", label="Granite 4.0 Response", lines=21, interactive=False, elem_id="output")
                flag_btn = gr.Button("Flag", variant="secondary")

    # actions
    submit_btn.click(generate, inputs=[prompt, max_tokens, temperature, top_p], outputs=[output])
    clear_btn.click(lambda: "", None, prompt)

demo.launch(share=True, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
