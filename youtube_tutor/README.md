#  YouTube Tutor AI

An AI-powered multi-video learning assistant that:
- Ingests YouTube videos and stores their transcripts as vectors
- Clusters multiple videos by topic automatically
- Answers questions with **exact timestamp citations** so you can jump to the source

##  Features
- **Multi-Video Knowledge Base** — paste multiple links; all are searchable together
- **Timestamp-Aware Retrieval** — every answer links back to the exact video moment
- **Topic Clustering** — videos auto-grouped by topic using embeddings + KMeans
- **Persistent Storage** — ChromaDB persists across sessions; no re-ingestion needed
- **Chat History** — multi-turn conversation with memory

##  Project Structure
```
youtube_tutor/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── .env                          # Your API keys (not committed)
├── .env.example
├── utils/
│   ├── youtube_loader.py         # Transcript fetching + chunking
│   ├── vector_store.py           # ChromaDB operations
│   ├── topic_clustering.py       # KMeans topic clustering
│   └── qa_chain.py               # LLM Q&A with citations
├── components/
│   ├── video_ingestion.py        # Add video UI
│   ├── knowledge_base.py         # View stored videos UI
│   └── chat_interface.py         # Chat UI with source cards
└── data/
    └── vectorstore/              # ChromaDB persistent storage
```

##  Setup

### 1. Clone / Open project in VS Code

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# OR
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API Keys
```bash
cp .env.example .env
```
Edit `.env` and fill in:
```
OPENAI_API_KEY=sk-...
YOUTUBE_API_KEY=AIza...       # Optional but recommended
```

### 5. Run the app
```bash
streamlit run app.py
```

##  Getting API Keys

### GroqAI API Key (Required)
1. Go to https://platform.groqai.com/api-keys
2. Create new secret key
3. Paste into `.env`

### YouTube Data API Key (Optional — for video titles)
1. Go to https://console.cloud.google.com
2. Create project → Enable "YouTube Data API v3"
3. Create credentials → API Key
4. Paste into `.env`

> Without YouTube API key, video titles will show as "Video <ID>" but everything else works.
