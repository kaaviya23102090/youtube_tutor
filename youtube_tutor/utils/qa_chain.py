import os
from groq import Groq

def build_context_from_results(results):
    context_parts = []
    for i, r in enumerate(results):
        timestamp_link = f"{r['video_url']}&t={int(r['start_seconds'])}"
        context_parts.append(
            f"[Source {i+1}] Video: \"{r['video_title']}\"\n"
            f"Timestamp: {r['timestamp_str']} (Link: {timestamp_link})\n"
            f"Content: {r['text']}"
        )
    return "\n\n---\n\n".join(context_parts)

def answer_question(question, retrieved_results, chat_history):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context = build_context_from_results(retrieved_results)

    system_prompt = """You are an intelligent YouTube Tutor Assistant.
Answer questions based ONLY on the provided video transcript context.

Rules:
1. Always cite the exact timestamp and video title when referencing content.
   Format: [Video: "Title" at 4:32](YouTube link with timestamp)
2. If information comes from multiple videos, attribute each part clearly.
3. If answer not found, say: "I couldn't find this in the provided videos."
4. Be concise but thorough. Use bullet points for multi-part answers.
5. End with a "📍 Sources" section listing all videos and timestamps used."""

    messages = [{"role": "system", "content": system_prompt}]
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Context from YouTube Videos:\n{context}\n\nQuestion: {question}"
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # ✅ Free Groq model
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
    )
    return response.choices[0].message.content

def generate_video_summary(video_title, chunks):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    sample_text = " ".join([c["text"] for c in chunks[:20]])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # ✅ Free Groq model
        messages=[
            {"role": "system", "content": "You summarize YouTube video transcripts clearly."},
            {"role": "user", "content": f"""Summarize this YouTube video transcript.
Video Title: {video_title}

Transcript:
{sample_text}

Provide:
1. A 1-sentence overview
2. 3-4 key topics covered
3. Who would benefit from watching this"""}
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content