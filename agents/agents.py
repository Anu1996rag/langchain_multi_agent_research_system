
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.tools import web_search, scrape_url


load_dotenv()

llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=1000,
        temperature=0
    )


def search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )


def reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


 # Writer Chain
writer_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "Act as an Expert Research Writer. You possess advanced skills in deep literature synthesis, "
        "objective data analysis, and technical communication. Your writing style is objective, "
        "highly analytical, clear, and engaging."
    ),
    (
        "human",
        """I need a comprehensive, detailed research report based on a specific topic and its accompanying data. 
Your primary goal is to transform raw information and gathered research into a highly structured, 
clear, and insightful narrative that uncovers meaningful trends and strategic takeaways.

[INPUT DATA]
- Topic: {topic}
- Gathered Research / Key Facts: {research_data}

[REPORT STRUCTURE]
Please organize the detailed report using the following markdown structure:
1. Executive Summary: A high-level overview of the topic, key findings, and the bottom-line significance.
2. Introduction & Background: Contextualize the topic and outline why this research matters.
3. Core Findings & Analysis: Group the gathered research into 2-3 logical thematic subsections. Explain *what* the data says and *why* it matters.
4. Strategic Implications/Insights: Synthesize the findings to offer deeper, actionable insights or future outlooks.
5. Conclusion: A definitive summary of the report's main takeaways.
6. Sources: List all the URLs found in the research.

[CONSTRAINTS & QUALITY STANDARDS]
- Direct & Precise: Eliminate fluff or generic filler text. Focus entirely on evidence-backed analysis.
- Scannability: Use bold text for key metrics and clear section headers to ensure readability.
- Grounded in Fact: Rely strictly on the research provided above. Do not hallucinate external metrics or claims.
- Tone: Maintain a professional, objective, and authoritative tone throughout."""
    )
])

# 3. Create the Runnable Chain using LCEL (LangChain Expression Language)
# The pipeline pipes: Inputs -> Prompt Template -> LLM Model -> String Parser
writer_chain = writer_prompt_template | llm | StrOutputParser()



# Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Act as an Elite Academic Peer Reviewer and Methodological Research Critic. Your core expertise "
        "lies in identifying logical fallacies, assessing the strength of empirical evidence, "
        "uncovering data biases, and identifying blind spots in analytical reporting. Your tone is "
        "constructive, highly rigorous, objective, and intellectually demanding."
    ),
    (
        "human",
        """You have been provided with a newly drafted research report. Your task is to conduct a meticulous, 
line-by-line critical evaluation of the report. Your primary goal is not to praise the text, but to find 
hidden flaws, unbacked assumptions, and areas where the analysis fails to fully support its conclusions.

[INPUT DATA]
- Draft Report to Review: 
{report}

[CRITIQUE STRUCTURE]
Please organize your critical evaluation using the following markdown structure:
1. Methodological & Logical Integrity: Evaluate the strength of the arguments and logical leaps.
2. Evidence & Data Sufficiency: Identify specific claims that lack concrete data, citations, or quantitative backing.
3. Counter-Arguments & Blind Spots: Detail what alternative perspectives or edge cases the report failed to consider.
4. Language & Tone Audit: Point out any instances of subjective phrasing or speculative filler language.
5. Actionable Refinement List: Provide a bulleted checklist of exact revisions needed to make this report bulletproof.
6. Overall Score (out of 10)
7. Areas to Improve

[CONSTRAINTS & QUALITY STANDARDS]
- Constructive Friction: Explain *why* a gap exists and *what* kind of data is missing to fix it.
- Direct Citations: Quote a snippet or reference the specific section of the text when pointing out a flaw.
- Zero Fluff: Jump straight into the critique without introductory pleasantries."""
    )
])

# Create the independent critic chain
critic_chain = critic_prompt | llm | StrOutputParser()
