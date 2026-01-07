import streamlit as st
import sqlite3
import tempfile
from langchain_core.messages import AIMessage
from SQL_AGENT1 import get_langgraph_sql_agent

st.set_page_config(page_title="LangGraph SQL Agent", layout="wide")

st.markdown("<h2 style='text-align: center;'>🧠 LangGraph SQL Agent (DB Upload)</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your SQLite `.db` file", type=["db"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
        tmp_file.write(uploaded_file.read())
        db_path = tmp_file.name

    st.success("✅ Database uploaded!")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize session state for outputs if not present
    if "agent_output" not in st.session_state:
        st.session_state.agent_output = ""
    if "agent_query" not in st.session_state:
        st.session_state.agent_query = ""
    if "eval_result" not in st.session_state:
        st.session_state.eval_result = None
    if "eval_error" not in st.session_state:
        st.session_state.eval_error = ""

    # Sidebar: Evaluation
    st.sidebar.header("🧪 Evaluation")
    user_sql_query = st.sidebar.text_area(
        "Enter a SQL query to evaluate agent's performance",
        height=150,
        placeholder="SELECT * FROM albums WHERE artist = 'Queen';",
    )
    if st.sidebar.button("Run Evaluation Query") and user_sql_query.strip():
        try:
            cursor.execute(user_sql_query)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description] if cursor.description else []
            st.session_state.eval_result = (rows, col_names)
            st.session_state.eval_error = ""
        except Exception as e:
            st.session_state.eval_error = str(e)
            st.session_state.eval_result = None

    if st.session_state.eval_error:
        st.sidebar.error(f"SQL Error: {st.session_state.eval_error}")
    elif st.session_state.eval_result:
        rows, col_names = st.session_state.eval_result
        if rows:
            st.sidebar.dataframe(rows, width=350, height=250)
        else:
            st.sidebar.info("No results found for the query.")

    # Main app: Agent interaction
    agent = get_langgraph_sql_agent(db_path)

    user_query = st.text_input("Ask a question to the agent", value="What is the title of the albums by artist 'Queen'?")

    if st.button("Submit") and user_query:
        st.session_state.agent_output = ""
        st.session_state.agent_query = ""
        output_placeholder = st.empty()
        sql_query_placeholder = st.empty()

        try:
            for step in agent.stream(
                {"messages": [{"role": "user", "content": user_query}]},
                stream_mode="values",
            ):
                for msg in step["messages"]:
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                if "query" in tc["args"]:
                                    st.session_state.agent_query = tc["args"]["query"]
                        st.session_state.agent_output += msg.content
                # Update placeholders inside the loop without rerun
                output_placeholder.markdown(st.session_state.agent_output)
                sql_query_placeholder.code(st.session_state.agent_query, language="sql")
        except Exception as e:
            st.error(f"⚠️ Error during agent execution: {e}")

