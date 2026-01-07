# sql_agent.py

import os
from typing import Literal
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()

def get_langgraph_sql_agent(db_path: str):
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatOpenAI(
        openai_api_base="https://llmfoundry.straive.com/openai/v1/",
        openai_api_key=f'{os.environ["LLMFOUNDRY_TOKEN"]}:my-test-project',
        model="gpt-4o-mini",
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    get_schema_node = ToolNode([get_schema_tool], name="get_schema")
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    run_query_node = ToolNode([run_query_tool], name="run_query")

    generate_query_system_prompt = f"""
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {db.dialect} query to run,
    then look at the results of the query and return the answer. 

    Never query all columns; only the relevant ones.
    DO NOT perform any DML (INSERT, UPDATE, DELETE).
    """

    check_query_system_prompt = f"""
    You are a SQL expert with attention to detail.
    Double-check this {db.dialect} query for issues like:
    - NOT IN with NULLs
    - UNION vs UNION ALL
    - BETWEEN usage
    - Data type mismatches
    - Identifier quoting
    - Function arguments
    - Casting
    - Join column correctness

    If there's an error, rewrite. Otherwise, return the original.
    """

    def list_tables(state: MessagesState):
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])

        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        tool_message = list_tables_tool.invoke(tool_call)
        # response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message]}

    def call_get_schema(state: MessagesState):
        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate_query(state: MessagesState):
        system_message = {"role": "system", "content": generate_query_system_prompt}
        llm_with_tools = llm.bind_tools([run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def check_query(state: MessagesState):
        system_message = {"role": "system", "content": check_query_system_prompt}
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        print("Response",response)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
        last_message = state["messages"][-1]
        return "check_query" if last_message.tool_calls else END

    builder = StateGraph(MessagesState)
    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(run_query_node, "run_query")

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")

    return builder.compile()
