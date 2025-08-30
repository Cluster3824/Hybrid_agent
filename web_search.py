import os
import re
import wikipedia
from typing import Dict
from datetime import datetime
import praw
import gradio as gr
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from serpapi import GoogleSearch

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "a61a4486857efe8d02b2533a5d1677e39d8dd78c0db751c4d47f2f60d80a8d86")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDEY_weJmWYsjkQmwRqOI4hGAo5-Gxik_0")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "Dvwiz3ue2NMhqGhDCEK1nQ")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "xa3jUxc2LxGsA8uaFnWf6X4xvVlRBg")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "Hybrid_agent")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "ollama").lower()
if DEFAULT_LLM not in {"ollama", "gemini"}:
    DEFAULT_LLM = "ollama"

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    check_for_async=False,
)

ollama_llm = ChatOllama(model=OLLAMA_MODEL)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

def parse_freshness(query: str):
    mapping = {
        r"\btoday\b|\blast 24 hours\b": ("d", "day"),
        r"\byesterday\b": ("d", "day"),
        r"\blast week\b|\bpast week\b": ("w", "week"),
        r"\blast month\b|\bpast month\b": ("m", "month"),
        r"\blast year\b|\bpast year\b": ("y", "year"),
    }
    for pattern, (w, r) in mapping.items():
        if re.search(pattern, query, re.IGNORECASE):
            return re.sub(pattern, "", query, flags=re.IGNORECASE).strip(), w, r
    return query, "w", "week"

def web_search(query: str, max_results: int = 5, freshness: str = "w") -> str:
    try:
        if not SERPAPI_KEY:
            return "[Google Search Skipped: SERPAPI_KEY not set]"
        params = {"q": query, "num": max_results, "api_key": SERPAPI_KEY}
        freshness_map = {"d": "qdr:d", "w": "qdr:w", "m": "qdr:m", "y": "qdr:y"}
        if freshness in freshness_map:
            params["tbs"] = freshness_map[freshness]

        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])
        formatted = [
            f"{r.get('title','').strip()} - {r.get('link','').strip()}\n{r.get('snippet','').strip()}"
            for r in results
        ]
        return "\n\n".join(formatted) or "No recent web results found."
    except Exception as e:
        return f"[Google Search Error: {e}]"

def reddit_search(query: str, limit: int = 5, freshness: str = "week") -> str:
    try:
        freshness_map = {"d": "day", "w": "week", "m": "month", "y": "year"}
        time_filter = freshness_map.get(freshness, "week")

        results = []
        for submission in reddit.subreddit("all").search(query, limit=limit, time_filter=time_filter):
            snippet = (submission.selftext or "").strip()
            snippet = snippet[:500] + "…" if len(snippet) > 500 else snippet
            results.append(
                f"Title: {submission.title}\n"
                f"Subreddit: r/{submission.subreddit.display_name}\n"
                f"Score: {submission.score} | Comments: {submission.num_comments}\n"
                f"URL: {submission.url}\n"
                f"Text: {snippet or 'No text'}"
            )
        return "\n\n".join(results) or "No recent Reddit results found."
    except Exception as e:
        return f"[Reddit Search Error: {e}]"

def wikipedia_search(query: str, sentences: int = 3) -> str:
    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(query, sentences=sentences)
    except wikipedia.DisambiguationError as e:
        return f"Multiple results found: {', '.join(e.options[:5])}..."
    except wikipedia.PageError:
        return "No Wikipedia page found for this query."
    except Exception as e:
        return f"[Wikipedia Search Error: {e}]"

def tools_node(state: Dict, use_web=True, use_reddit=True, use_wiki=True) -> Dict:
    query, web_freshness, reddit_freshness = parse_freshness(state["input"])
    today = datetime.now().strftime("%B %d, %Y")

    web_results = web_search(query, max_results=5, freshness=web_freshness) if use_web else ""
    reddit_results = reddit_search(query, limit=5, freshness=reddit_freshness) if use_reddit else ""
    wiki_result = wikipedia_search(query) if use_wiki else ""

    return {
        "input": f"Date: {today}\n\n{query}",
        "web_refs": {f"Web{i+1}": line for i, line in enumerate(web_results.split('\n\n')) if line.strip()},
        "reddit_refs": {f"Reddit{i+1}": line for i, line in enumerate(reddit_results.split('\n\n')) if line.strip()},
        "wiki_refs": {"Wikipedia": wiki_result} if wiki_result else {},
    }

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert research assistant. Use BOTH the summarized chat history and the last few raw turns "
     "plus references (Web, Reddit, Wikipedia) to give a detailed, contextual answer. "
     "Do NOT include raw reference links."),
    ("system", "Summarized History:\n{history_summary}"),
    ("system", "Recent Chat:\n{chat_history}"),
    ("human", "Question: {input}\n\nAnswer in detail without including references.")
])

def run_llm(chain: LLMChain, state: Dict, memory: ConversationBufferWindowMemory,
            summary_memory: ConversationSummaryMemory, model_choice: str) -> Dict:
    try:
        chat_history_list = memory.load_memory_variables({}).get("chat_history", [])
        chat_history_text = "\n".join([
            f"User: {msg.content}" if msg.type == "human" else f"Assistant: {msg.content}"
            for msg in chat_history_list
        ]) if chat_history_list else ""

        history_summary_text = summary_memory.load_memory_variables({}).get("history_summary", "")

        inputs = {
            "input": state["input"],
            "chat_history": chat_history_text,
            "history_summary": history_summary_text
        }
        answer = chain.run(inputs)
    except Exception as e:
        if model_choice.lower() == "ollama":
            try:
                fallback_chain = LLMChain(llm=gemini_llm, prompt=prompt, memory=memory, verbose=False)
                answer = fallback_chain.run(inputs)
                answer = f"(Ollama unavailable, switched to Gemini)\n\n{answer}"
            except Exception as e2:
                answer = f"[LLM Error: {e} | Gemini Fallback Error: {e2}]"
        else:
            answer = f"[LLM Error: {e}]"

    memory.save_context({"input": state["input"]}, {"output": answer})
    summary_memory.save_context({"input": state["input"]}, {"output": answer})
    return {"answer": answer}

def chat_fn(user_message, history, model_choice, memory, summary_memory, use_web, use_reddit, use_wiki):
    if not history:
        history = []

    casual_responses = {
        "hi": "Hey there! How can I assist you today?",
        "hello": "Hello! How’s your day going?",
        "hey": "Hi! What can I help you with?",
        "how are you": "I'm doing great, thanks! How about you?",
        "good morning": "Good morning! What can I do for you today?",
        "good evening": "Good evening! How can I assist?"
    }
    lower_msg = user_message.strip().lower()
    for phrase, reply in casual_responses.items():
        if phrase in lower_msg and len(user_message.split()) <= 5:
            history.append((user_message, reply))
            return history, "", memory, summary_memory

    if "time" in lower_msg:
        current_time = datetime.now().strftime("%I:%M %p")
        history.append((user_message, f"The current time is {current_time}."))
        return history, "", memory, summary_memory

    if "remind me" in lower_msg:
        history.append((user_message, "Got it! I'll note that down (simulation – no persistent reminders yet)."))
        return history, "", memory, summary_memory

    if memory is None:
        memory = ConversationBufferWindowMemory(
            k=5, memory_key="chat_history", input_key="input", return_messages=True
        )
    if summary_memory is None:
        summary_memory = ConversationSummaryMemory(
            llm=gemini_llm, memory_key="history_summary", input_key="input"
        )

    ollama_chain = LLMChain(llm=ollama_llm, prompt=prompt, memory=memory, verbose=False)
    gemini_chain = LLMChain(llm=gemini_llm, prompt=prompt, memory=memory, verbose=False)
    chain = gemini_chain if model_choice.lower() == "gemini" else ollama_chain

    state = tools_node({"input": user_message}, use_web, use_reddit, use_wiki)
    result = run_llm(chain, state, memory, summary_memory, model_choice)
    answer = result.get("answer", "No answer.")

    web_refs = state.get("web_refs", {})
    reddit_refs = state.get("reddit_refs", {})
    wiki_refs = state.get("wiki_refs", {})

    references = ""
    if web_refs:
        references += "Web References:\n" + "\n".join([f"{k}: {v}" for k, v in web_refs.items()]) + "\n\n"
    if reddit_refs:
        references += "Reddit References:\n" + "\n".join([f"{k}: {v}" for k, v in reddit_refs.items()]) + "\n\n"
    if wiki_refs:
        references += "Wikipedia:\n" + "\n".join([f"{k}: {v}" for k, v in wiki_refs.items()])

    history.append((user_message, answer))
    return history, references, memory, summary_memory

with gr.Blocks(title="Hybrid Research Agent Chatbot", theme="soft") as iface:
    gr.Markdown("## Hybrid Research Agent\nUses Browser, Reddit, Wikipedia\n(With Summarized + Limited Memory)")

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["ollama", "gemini"],
            value=DEFAULT_LLM,
            label="Choose LLM",
            interactive=True
        )
        use_web = gr.Checkbox(value=True, label="Use Web Search")
        use_reddit = gr.Checkbox(value=True, label="Use Reddit")
        use_wiki = gr.Checkbox(value=True, label="Use Wikipedia")

    with gr.Row():
        with gr.Column(scale=3):
            chatbox = gr.Chatbot(label="Chat", height=400, bubble_full_width=False, avatar_images=("🧑", "🤖"))
            user_input = gr.Textbox(placeholder="Ask a question...", label="Your Question", lines=2)
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                export_btn = gr.Button("📥 Export Chat", variant="secondary")
        with gr.Column(scale=2):
            references_box = gr.Textbox(label="References", lines=20, interactive=False)

    session_history = gr.State([])
    session_memory = gr.State(None)
    session_summary = gr.State(None)

    def submit_message(user_message, history, model_choice, memory, summary_memory, use_web, use_reddit, use_wiki):
        return chat_fn(user_message, history, model_choice, memory, summary_memory, use_web, use_reddit, use_wiki)

    def clear_chat():
        return [], "", None, None

    def export_chat(history):
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for u, a in history:
                f.write(f"User: {u}\nBot: {a}\n\n")
        return filename

    user_input.submit(
        submit_message,
        inputs=[user_input, session_history, model_choice, session_memory, session_summary, use_web, use_reddit, use_wiki],
        outputs=[chatbox, references_box, session_memory, session_summary]
    )
    send_btn.click(
        submit_message,
        inputs=[user_input, session_history, model_choice, session_memory, session_summary, use_web, use_reddit, use_wiki],
        outputs=[chatbox, references_box, session_memory, session_summary]
    )
    clear_btn.click(
        clear_chat,
        inputs=[],
        outputs=[chatbox, references_box, session_memory, session_summary]
    )
    export_btn.click(
        export_chat,
        inputs=[session_history],
        outputs=[]
    )

if __name__ == "__main__":
    iface.launch()
