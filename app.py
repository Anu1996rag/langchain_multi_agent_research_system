import streamlit as st
from agents.agents import search_agent, reader_agent, writer_chain, critic_chain

def run_pipeline_ui(topic):
    state = {}
    
    # --- Step 1: Search Agent ---
    st.subheader("🔍 Step 1: Search Agent")
    with st.spinner("Searching for recent and detailed information..."):
        try:
            srch_agent = search_agent()
            search_results = srch_agent.invoke({
                "messages": [
                    ("user", f"Find recent, relevant and detailed information about : {topic}")
                ]
            })
            state["search_results"] = search_results['messages'][-1].content
            st.success("Search completed!")
            with st.expander("View Raw Search Results", expanded=False):
                st.text(state["search_results"])
        except Exception as e:
            st.error(f"Error in Search Agent: {e}")
            return None

    # --- Step 2: Extract Agent ---
    st.subheader("📄 Step 2: Extract Agent")
    with st.spinner("Selecting relevant URL and scraping content..."):
        try:
            rd_agent = reader_agent()
            extracted_results = rd_agent.invoke({
                "messages": [
                    ("user", f"Based on following searched results about {topic} "
                             f"Pick the most relevant URL and scrape the content. "
                             f"Search Results : \n {state['search_results'][:1000]}")
                ]
            })
            state["scraped_content"] = extracted_results['messages'][-1].content
            st.success("Scraping completed!")
            with st.expander("View Scraped Content", expanded=False):
                st.text(state["scraped_content"])
        except Exception as e:
            st.error(f"Error in Extract Agent: {e}")
            return None

    # --- Step 3: Writer Agent ---
    st.subheader("✍️ Step 3: Writer Agent")
    with st.spinner("Compiling and generating research report..."):
        try:
            research_data = (
                f"Search Results : \n {state['search_results']}"
                f"Detailed Content: \n {state['scraped_content']}"
            )
            state["report"] = writer_chain.invoke({
                "topic": topic,
                "research": research_data
            })
            st.success("Report generation completed!")
        except Exception as e:
            st.error(f"Error in Writer Agent: {e}")
            return None

    # --- Step 4: Critic Agent ---
    st.subheader("⚖️ Step 4: Critic Agent")
    with st.spinner("Reviewing and evaluating the report..."):
        try:
            state["critics"] = critic_chain.invoke({
                "report": state["report"]
            })
            st.success("Critic review completed!")
        except Exception as e:
            st.error(f"Error in Critic Agent: {e}")
            return None

    return state

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Multi-Agent Research System", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent Research Assistant")
st.markdown("This application uses sequential LangChain agents to search, extract, write, and critique information on any given topic.")

topic_input = st.text_input("Enter your research topic:", placeholder="e.g., Quantum Computing advancements")

if st.button("Start Research Pipeline", type="primary"):
    if not topic_input.strip():
        st.warning("Please enter a valid topic before running the pipeline.")
    else:
        st.divider()
        pipeline_results = run_pipeline_ui(topic_input)
        
        if pipeline_results:
            st.divider()
            st.header("🎯 Final Pipeline Outputs")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 Generated Report")
                st.markdown(pipeline_results["report"])
                
            with col2:
                st.subheader("🔬 Critic Feedback")
                st.markdown(pipeline_results["critics"])
