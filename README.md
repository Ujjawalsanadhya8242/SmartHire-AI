# SmartHire AI 🚀

SmartHire AI is a GPU-accelerated resume parser powered by advanced LLMs to extract structured data (name, email, phone, skills, experience, education) from resumes in PDF format. It enables hiring teams to automate and optimize candidate screening with high accuracy and speed.

## 🌟 Features

- ✅ Fast and accurate resume parsing
- 🤖 Powered by OpenHermes 2.5 - Mistral 7B LLM
- ⚡ GPU-accelerated inference with 8-bit quantization
- 📄 PDF resume upload support
- 🌐 Modern web interface (Tailwind CSS + Jinja2)
- 🔗 Public access via Ngrok tunnel
- 🔍 Structured JSON output for easy downstream use

## 🧠 Objective

To revolutionize the recruitment process by leveraging LLMs to automatically extract structured information from resumes, reducing manual screening effort and enabling faster decision-making.

## 📦 Tech Stack

- **Backend:** FastAPI
- **Model:** OpenHermes 2.5 - Mistral 7B (HuggingFace)
- **Frontend:** TailwindCSS + Jinja2 Templates
- **PDF Extraction:** `pdfplumber`
- **Inference Acceleration:** `bitsandbytes`, 8-bit quantization
- **Tunneling (optional):** Ngrok

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/smarthire-ai.git
cd smarthire-ai
