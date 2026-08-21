from agents.agents import search_agent, reader_agent, writer_chain, critic_chain

def run_pipeline(topic):

    state = {}

    srch_agent = search_agent()
    search_results = srch_agent.invoke({
        "messages": [
            ("user", f"Find recent, relevant and detailed information about : {topic}")
        ]
    })
    state["search_results"] = search_results['messages'][-1].content

    rd_agent = reader_agent()
    extracted_results = rd_agent.invoke({
        "messages": [
            ("user", f"Based on following searched results about {topic}"
                     f"Pick the most relevant URL and scrape the content."
                     f"Search Results : \n {state['search_results'][:1000]}")
        ]
    })
    state["scraped_content"] = extracted_results['messages'][-1].content

    research_data = (
        f"Search Results : \n {state['search_results']}"
        f"Detailed Content: \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke(
        {
            "topic": topic,
            "research": research_data
        }
    )

    state["critics"] = critic_chain.invoke(
        {
            "report": state["report"]
        }
    )
