# 🧠 RAG-Based Teaching AI Assistant  
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) 
[![Colab](https://img.shields.io/badge/Run%20on-Colab-orange.svg)](https://colab.research.google.com/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Vosk](https://img.shields.io/badge/Speech--to--Text-VOSK-lightgrey.svg)](https://alphacephei.com/vosk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🎓 A complete Retrieval-Augmented Generation (RAG) pipeline that transforms video lectures into an **AI-powered teaching assistant**, using **offline speech recognition (Vosk)** and **context-aware Q&A** through **Groq** or **Gemini**.

---

## 🚀 Overview

This project turns your **teaching videos** into an **intelligent assistant** capable of answering questions based on their content.  
It combines **offline transcription**, **semantic embeddings**, and **LLM reasoning** to make learning interactive and accessible anywhere — even offline.

---

## 🧩 Key Features

✅ **Video → Audio → Text Pipeline**  
Convert `.mp4` lectures into `.mp3` and transcribe them using **Vosk** (offline ASR).

✅ **Clean JSON Output**  
Automatic cleaning of transcripts for easy processing and embedding.

✅ **Embeddings + FAISS Index**  
Generate semantic embeddings with `SentenceTransformers` and store them in a **FAISS** vector database for lightning-fast retrieval.

✅ **RAG with Groq or Gemini**  
Retrieve relevant transcript chunks and answer questions with context-aware reasoning using **Groq** or **Google Gemini** APIs.

✅ **Interactive Console**  
Ask natural-language questions directly inside the notebook.

---

## 🧱 Project Structure

```

📦 RAG-Based-AI-Teaching-Assistant/
├── videos/               # Original lecture videos (.mp4)
├── audios/               # Converted audio files (.mp3)
├── jsons/                # Raw Vosk transcripts
├── jsons/clean/          # Cleaned transcripts
├── vosk_model/           # Offline Vosk model directory
├── embeddings.joblib     # Stored embeddings for retrieval
├── RAG_Teaching_AI_Assistant.ipynb  # Main Colab notebook
└── README.md             # Project documentation

````

---

## ⚙️ Setup & Installation

### 1️⃣ Clone or open in Google Colab
```bash
git clone https://github.com/<your-username>/RAG-AI-Teaching-Assistant.git
cd RAG-AI-Teaching-Assistant
````

or open directly in Colab and run all cells sequentially.

### 2️⃣ Install dependencies

```bash
pip install vosk soundfile pydub tqdm ffmpeg-python sentence-transformers joblib pandas faiss-cpu groq google-generativeai
```

Ensure that **ffmpeg** is installed (Colab auto-installs it).

---

## 🔄 Workflow Summary

| Step  | Description                              | Notebook Cell |
| ----- | ---------------------------------------- | ------------- |
| **1** | Install & configure Vosk                 | Cell 1        |
| **2** | Mount Google Drive & import project      | Cell 2        |
| **3** | Ensure required folders exist            | Cell 3        |
| **4** | Convert videos → MP3                     | Cell 4        |
| **5** | Transcribe audio → JSON (offline)        | Cell 5        |
| **6** | Clean & preprocess transcripts           | Cell 6        |
| **7** | Generate & store embeddings              | Cell 7        |
| **8** | Build FAISS index + RAG QA (Groq/Gemini) | Cell 8        |

---

## 🔑 API Configuration

Set up one of the following API keys before running **Cell 8**:

### 🧠 Using Groq

```python
import os
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
```

### 🌐 Using Gemini

```python
import os
os.environ["GEMINI_API_KEY"] = "your_gemini_api_key"
```

---

## 💬 Example Interaction

```
🔌 Provider: GROQ
❓ Your question: What is a black hole?

🔎 Retrieved Context:
 [1] Black holes are regions in space where gravity is so strong...

💬 Answer:
 A black hole is a region in space where the gravitational pull is so intense that nothing, not even light, can escape.
```

---

## 🧠 Tech Stack

| Category               | Tools / Libraries                     |
| ---------------------- | ------------------------------------- |
| **Language**           | Python 3.10+                          |
| **Environment**        | Google Colab                          |
| **Speech-to-Text**     | [Vosk](https://alphacephei.com/vosk/) |
| **Audio Processing**   | FFmpeg, Pydub                         |
| **Embeddings**         | Sentence-Transformers                 |
| **Vector Search**      | FAISS                                 |
| **LLM API**            | Groq / Google Gemini                  |
| **Data Serialization** | JSON, Joblib, Pandas                  |

---

## 📈 Future Enhancements

* 🔊 Integrate **Whisper** or **OpenAI** ASR as optional transcription backend
* 🌍 Add **multilingual** transcription and embedding support
* 🧩 Build a **Gradio/Streamlit UI** for web-based chat
* 🧮 Summarize or cluster lectures by topic
* 💾 Integrate vector-store persistence (e.g., ChromaDB)

---

## 👨‍💻 Author

**Mohammed Enayatullah Safwan**
🎓 *National Institute of Technology, Durgapur*
📧 [LinkedIn](https://www.linkedin.com/in/mdsafwan86/) | [GitHub](https://github.com/safwanig)


> ✨ *“Transform your lectures into knowledge-driven conversations — powered by AI.”*


