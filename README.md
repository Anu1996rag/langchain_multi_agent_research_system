# 🔬 AI Research Multi-Agent Pipeline

An intuitive, interactive **Multi-Agent Research System** built with **LangChain** and **Streamlit**. This application automates the process of information gathering by coordinating four specialized agents to search the web, scrape deep content, draft a comprehensive report, and critically evaluate the output—all running **100% locally** using **Ollama** and **ChatOllama**.

---

## 🛠️ System Architecture & Workflow

The pipeline operates sequentially, passing data from one specialized agent or chain to the next:

```
[User Input: Topic] 
       │
       ▼
┌────────────────────────┐
│  🔎 1. Search Agent    │ ──► Discovers recent, relevant web URLs & snippets
└────────────────────────┘
       │
       ▼
┌────────────────────────┐
│  📄 2. Extract Agent   │ ──► Dynamically picks the best URL & scrapes full content
└────────────────────────┘
       │
       ▼
┌────────────────────────┐
│  ✍️ 3. Writer Chain    │ ──► Synthesizes search & scraped data into a structured report
└────────────────────────┘
       │
       ▼
┌────────────────────────┐
│  ⚖️ 4. Critic Chain    │ ──► Evaluates the report for accuracy, gaps, and improvements
└────────────────────────┘
```

---

## 🚀 Getting Started

Follow these steps to set up Ollama locally, install dependencies, and launch the interactive UI.

### 1. Set Up Ollama Locally

Since this system relies on locally hosted models via `ChatOllama`, you must have the Ollama service running on your machine.

1. **Download & Install Ollama:**
   * Go to the official website: [Ollama.ai](https://ollama.com/)
   * Download and install the version compatible with your OS (Mac, Linux, or Windows).

2. **Pull Your Preferred Local Model:**
   Open your terminal/command prompt and download an LLM (e.g., `llama3`, `mistral`, or `phi3`). For example, to pull Llama 3:
   ```bash
   ollama pull llama3
   ```

3. **Start the Ollama Server:**
   Keep the Ollama background service running. Usually, launching the app or running any model starts it automatically, but you can explicitly serve it via:
   ```bash
   ollama serve
   ```

---

### 2. Installation & Setup

1. **Clone or Setup Your Project Directory:**
   Ensure your folder structure looks like this:
   ```text
   ├── agents/
   │   ├── __init__.py
   │   └── agents.py       # Contains search_agent, reader_agent, writer_chain, critic_chain
   ├── pipeline.py         # Your core pipeline execution script
   ├── app.py              # Streamlit Web UI script
   └── README.md           # This file!
   ```

2. **Install Required Dependencies:**
   Run the following command to install the required Python packages:
   ```bash
   pip install streamlit langchain langchain-community
   ```

---

### 3. Running the Application

You can interact with the system in two ways: via the CLI terminal pipeline or the rich web interface.

#### 🌐 Option A: Interactive Streamlit Web UI (Recommended)
Launch the responsive browser interface to visualize agent execution states in real-time:
```bash
streamlit run app.py
```

#### 💻 Option B: Core Python Pipeline Execution
To execute the pipeline directly in your terminal, make sure your script invokes the core runner function:
```bash
python pipeline.py
```

---

## 🎨 Interactive Web UI Tour

When you launch the Streamlit app, you will experience a highly interactive workspace:

* **⚡ Real-Time Status Indicators:** Watch spinning status tickers as each agent wakes up, executes, and passes state weights down the line.
* **📦 Accordion Data Toggles:** Expandable text cards let you peer directly into raw `Search Results` and `Scraped Web Content` chunks without overwhelming your clean viewport layout.
* **📊 Dual-Column Markdown View:** Review the synthesized **Agent Draft Report** side-by-side with the critical **Validation & Feedback** notes provided by the Critic agent.

---

## 🔧 Model Configuration Notice
Make sure your initialization logic inside `agents/agents.py` correctly targets your locally hosted Ollama instance. For example:
```python
from langchain_community.chat_models import ChatOllama

# Example configuration within your agent definitions
llm = ChatOllama(
    model="llama3", 
    temperature=0.3
)
```
