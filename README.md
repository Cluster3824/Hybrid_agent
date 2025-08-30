# Hybrid_agent

An AI-powered research assistant that integrates real-time web search, Reddit, and Wikipedia with hybrid LLM support (Ollama + Gemini).
It combines external knowledge sources with contextual memory to provide detailed, up-to-date, and conversational responses.

✨ Features

🔍 Real-time Web Search → Uses Google Search (via SerpAPI) with freshness filters (day, week, month, year).

📢 Reddit Integration → Fetches latest community discussions, metadata (upvotes, comments), and snippets.

📚 Wikipedia Summaries → Retrieves concise explanations with disambiguation handling.

🤖 Hybrid LLM Choice → Switch between Ollama (LLaMA 3.2) and Gemini (Google Generative AI) dynamically.

🧠 Contextual Memory →

ConversationBufferWindowMemory → Short-term history.

ConversationSummaryMemory → Summarized long-term history.

🎯 Reference Management → Structured references panel showing results from Web, Reddit, and Wikipedia.

💬 Casual Chat Features → Handles greetings, time queries, and reminders.

🌐 Interactive UI → Built with Gradio, supports chat export and clearing history.

⚙️ Tech Stack

Python 3.10+

LangChain

Gradio

Ollama (LLaMA 3.2)

Google Generative AI (Gemini)

PRAW (Reddit API)

SerpAPI (Google Search API)

Wikipedia API
