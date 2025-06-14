# # main.py
# from fastapi import FastAPI, File, UploadFile, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# import json
# import pdfplumber
# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     pipeline,
#     BitsAndBytesConfig
# )

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # --- MODEL LOADING ---
# model_id = "MohammadOthman/OpenHermes-2.5-Mistral-7B-Orca-DPO"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# if device == "cuda":
#     quant_config = BitsAndBytesConfig(load_in_8bit=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, quantization_config=quant_config, device_map="auto"
#     )
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, device_map="auto", torch_dtype=torch.float16
#     )

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# resume_parser = pipeline(
#     "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
# )

# # --- ENDPOINTS ---
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "data": None})

# @app.post("/process", response_class=HTMLResponse)
# async def process(request: Request, pdf_doc: UploadFile = File(...)):
#     with pdfplumber.open(pdf_doc.file) as pdf:
#         text = "\n".join(page.extract_text() or "" for page in pdf.pages)

#     model_input = (
#         "You are an expert resume parser. Given a resume, extract only the relevant fields and return a valid JSON with these keys: 'name', 'email', 'phone', 'education', 'skills', and 'experience'. Do not return any tips, commentary, or additional formatting. Each value should be concise and accurate. Structure experience and education as arrays of objects if needed."
#         f"{text}"
#     )

#     output = resume_parser(model_input, max_new_tokens=256)[0]["generated_text"]

#     try:
#         parsed = json.loads(output)
#     except json.JSONDecodeError:
#         parsed = {"error": "Could not parse JSON", "raw_output": output}

#     return templates.TemplateResponse("index.html", {"request": request, "data": parsed})



from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import pdfplumber
import torch
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- MODEL LOADING ---
model_id = "MohammadOthman/OpenHermes-2.5-Mistral-7B-Orca-DPO"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)

# --- HELPER TO EXTRACT CLEAN JSON ---
def extract_json(output_text):
    match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Could not parse JSON", "raw_output": output_text}


# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "data": None})
@app.post("/process", response_class=HTMLResponse)
async def process(request: Request, pdf_doc: UploadFile = File(...)):
    with pdfplumber.open(pdf_doc.file) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    prompt = (
        "You are an expert resume parser. Given a resume, extract only the relevant fields and return a valid JSON with these keys: "
        "'name', 'email', 'phone', 'education', 'skills', and 'experience'. "
        "Do not return any tips, commentary, or additional formatting. "
        "Each value should be concise and accurate. Structure experience and education as arrays of objects if needed.\n\n"
        "Resume:\n"
        f"{text.strip()}\n\n"
        "JSON:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # Get only the new tokens (the model's answer), not the input prompt
    output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Extract the JSON part using regex
    match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            parsed = {"error": "Could not parse JSON", "raw_output": output_text}
    else:
        parsed = {"error": "JSON not found", "raw_output": output_text}

    return templates.TemplateResponse("index.html", {"request": request, "data": parsed})


