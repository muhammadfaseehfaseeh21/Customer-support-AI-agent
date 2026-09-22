import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from tools import SupportRAGTool

def run_support_agent(user_query: str, api_key: str) -> str:
    # Set up LLM with requested open model endpoint
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        openai_api_key=api_key,
        temperature=0.2
    )

    # Initialize the RAG search tool
    rag_tool = SupportRAGTool(json_path="knowledge_base.json", openai_api_key=api_key)

    # Single CrewAI Agent
    support_agent = Agent(
        role="Customer Support Representative",
        goal="Provide clear, empathetic, and accurate responses to customer queries regarding order details, delivery statuses, and profiles.",
        backstory=(
            "You are a helpful and polite customer support agent for an e-commerce platform. "
            "You always double-check records using the Customer Knowledge Base Search tool "
            "to answer queries about Customer ID, order date, contact number, status, and shopping address accurately."
        ),
        tools=[rag_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    # Define Single Task
    support_task = Task(
        description=(
            f"Answer the customer's request: '{user_query}'\n\n"
            "Steps:\n"
            "1. Search the knowledge base for relevant Customer IDs, names, or order information.\n"
            "2. Extract essential details like Customer-ID, Order Date, Contact Number, Status, and Shopping Address.\n"
            "3. Format your final response in clean markdown with clear sections."
        ),
        expected_output="A friendly and structured customer support response presenting accurate account/order details.",
        agent=support_agent
    )

    # Assemble single-agent crew
    crew = Crew(
        agents=[support_agent],
        tasks=[support_task],
        process=Process.sequential
    )

    result = crew.kickoff()
    return str(result)
