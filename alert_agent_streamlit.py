import logging
import sys
import streamlit as st
import pandas as pd
import re
from langchain_community.utilities import SQLDatabase
from langchain.messages import AIMessage, HumanMessage
from send_mail import send_email
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from my_Sql_connection import sql_connection
from Openi_ai import build_llm
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from langgraph.graph import END, START, StateGraph, MessagesState
from typing import Literal
import ast
from datetime import datetime
import re
import pandas as pd
from sqlalchemy import text
from langchain.tools import tool
import ast
# Configure Streamlit page
st.set_page_config(
    page_title="SQL Alert Detection Agent",
    page_icon="🔔",
    layout="wide"
)




import re
import ast
from datetime import datetime

def safe_parse_result(raw):
    """
    Converts string result containing datetime.datetime(...) into 
    a safely literal_eval()-able dict.
    """

    # Step 1 — Replace datetime.datetime(...) with ISO string
    dt_pattern = r"datetime\.datetime\((.*?)\)"

    def dt_replacer(match):
        parts = [int(x.strip()) for x in match.group(1).split(",")]
        dt = datetime(*parts)
        return f"'{dt.strftime('%Y-%m-%d')}'"

    cleaned = re.sub(dt_pattern, dt_replacer, raw)

    # Step 2 — Safely parse with literal_eval
    return ast.literal_eval(cleaned)
def convert_to_dataframe(data_dict):
    import pandas as pd

    columns = data_dict["columns"]
    rows = data_dict["rows"]

    cleaned_rows = []
    for row in rows:
        cleaned_row = []

        for v in row:
            if isinstance(v, datetime):
                cleaned_row.append(v.strftime("%Y-%m-%d"))
            else:
                cleaned_row.append(v)

        cleaned_rows.append(cleaned_row)

    return pd.DataFrame(cleaned_rows, columns=columns)

# Setup logging
logger = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("app_log.txt", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'email_sent' not in st.session_state:
    st.session_state.email_sent = False
if 'alert_detected' not in st.session_state:
    st.session_state.alert_detected = False
if 'sql_query' not in st.session_state:
    st.session_state.sql_query = None
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'alert_message' not in st.session_state:
    st.session_state.alert_message = None


@st.cache_resource
def initialize_agent():
    """Initialize database connection and agent"""
    try:
        engine = sql_connection()
        logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        st.error(f"Database connection failed: {e}")
        return None

    try:
        llm = build_llm()
        logger.info("LLM initialization successful.")
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        st.error(f"LLM initialization failed: {e}")
        return None

    agent_obj = alert_agent(engine=engine, llm=llm)
    return build_graph(agent_obj)

class alert_agent:
    def __init__(self, engine, llm):
        self.db = SQLDatabase(engine=engine)
        logger.info("SQLDatabase object created.")

        self.llm = llm
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        logger.info(f"Toolkit initialized with {len(self.tools)} tools.")

        self.get_schema_tool = next(tool for tool in self.tools if tool.name == "sql_db_schema")
        self.run_query_tool = next(tool for tool in self.tools if tool.name == "sql_db_query")
        self.list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")
        logger.info("Key tools mapped.")

        self.get_schema_node = ToolNode([self.get_schema_tool], name="get_schema")

        @tool("sql_db_query_structured")
        def run_query_with_columns(query: str) -> dict:
            """Execute a SQL query and return structured columns + rows."""
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    rows = result.fetchall()
                    columns = result.keys()

                return {
                    "columns": list(columns),
                    "rows": [list(r) for r in rows]
                }

            except Exception as e:
                return {"error": str(e)}
        self.run_query_structured_tool = run_query_with_columns

        self.run_query_node = ToolNode([self.run_query_structured_tool], name="run_query")        
        logger.info("ToolNodes created for schema and query tools.")

    def list_available_tools(self):
        logger.info("Listing available tools:")
        for tool in self.tools:
            logger.info(f"{tool.name}: {tool.description}")

    def list_tables(self, state: MessagesState):
        logger.info("Executing list_tables tool.")
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "abc123",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = self.list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        logger.info(f"Tables retrieved: {tool_message.content}")
        return {"messages": [response]}

    def call_get_schema(self, state: MessagesState):
        logger.info("Executing call_get_schema tool.")
        llm_with_tools = self.llm.bind_tools([self.get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate_query(self, state: MessagesState):
        logger.info("Executing generate_query.")
        prompt = f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {self.db.dialect} query to run.

If the question doesn't specify specific columns, retrieve **all columns** from the relevant table(s).
If the question specifies particular columns (e.g., 'sales', 'product_name', etc.), only select those relevant columns.

Always limit your query to at most 5 results, unless the question explicitly asks for more.
Order the results by a relevant column to return the most interesting examples.

Always order the results by the most recent data (e.g., a timestamp, date, or other relevant chronological column) to ensure the newest entries appear first. Use order by time or data first then any other.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""
        system_message = {"role": "system", "content": prompt}
        llm_with_tools = self.llm.bind_tools([self.run_query_structured_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        # logger.info(f"the generated resposnse is this {response}")
        return {"messages": [response]}

    def check_query(self, state: MessagesState):
        logger.info("Executing check_query.")
        prompt = f"""
You are a SQL expert with strong attention to detail.
Double-check the {self.db.dialect} query for common mistakes, such as:
- Using NOT IN with NULL values
- UNION vs UNION ALL confusion
- BETWEEN for ranges
- Data type mismatches
- Proper quoting of identifiers
- Correct function arguments
- Proper casting
- Correct columns in joins

If errors exist, rewrite the query; otherwise, reproduce it.
You will execute the query after this validation.
"""
        system_message = {"role": "system", "content": prompt}
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = self.llm.bind_tools([self.run_query_structured_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        
        query_tool_call = response.tool_calls[0]
        query_tool_call = query_tool_call["args"]["query"]
        logger.info(f"Query generated --> {query_tool_call}")
        
        # Store SQL query in session state
        st.session_state.sql_query = query_tool_call
        
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    def detect_alerts(self, state: MessagesState):
        """
        Analyze query results and detect alerts using LLM.
        If alerts are found, send an email notification.
        """
        logger.info("Executing detect_alerts.")
        
        original_query = None
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                original_query = message.content
                break
        
        last_message = state["messages"][-1]
        query_result = last_message.content
        
        # Store query results in session state
        st.session_state.query_results = query_result
        # logger.info(f"checking query {query_result}")
        prompt = f"""
You are an Alert Analysis & Reporting System. 
Analyze the information below and generate a structured alert report **strictly based on the user's query**.

-----------------------------------------
USER QUERY:
{original_query}

QUERY RESULTS:
{query_result}
-----------------------------------------

Your task:
1. Identify alerts that are directly relevant to the user's query.
2. Only flag conditions that violate thresholds, expectations, or constraints implied by the user's query.
3. For each alert found, include:
   - **Alert Title**
   - **Description of the Issue**
   - **Evidence from Results**
   - **Why This Violates the User's Query**

Rules:
- DO NOT flag duplicate items.
- DO NOT flag small inconsistencies or irrelevant anomalies.
- DO NOT analyze anything outside the user's query intent.
- DO NOT include phrases like "End of Report" or "End".

Format the output as a clear, readable report.
"""


        system_message = {"role": "system", "content": prompt}
        response = self.llm.invoke([system_message])
        analysis_text = response.content.strip()
        
        # Store alert message in session state
        st.session_state.alert_message = analysis_text

        # EMAIL NOTIFICATION LOGIC
        if "NO_ALERTS DETECTED" not in analysis_text:
            st.session_state.alert_detected = True
            try:
                alert_subject = "Alert Detected in System"
                alert_body = f"{analysis_text}"
                recipient = st.session_state.get("email_recipient", "sahil.singla@algo8.ai")

                send_email(
                    alert_subject,
                    alert_body,
                    recipient
                )
                logger.info("Alert email sent successfully.")
                st.session_state.email_sent = True
            except Exception as e:
                logger.error(f"Failed to send alert email: {e}")
                st.session_state.email_sent = False
                st.error(f"Failed to send alert email: {e}")
        else:
            st.session_state.alert_detected = False
            st.session_state.email_sent = False

        alert_message = AIMessage(content=f"Alert Analysis:\n{analysis_text}")
        return {"messages": [alert_message]}

def should_continue(state: MessagesState) -> Literal["check_query", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("Tool calls detected, continuing to check_query.")
        return "check_query"
    else:
        logger.info("No tool calls detected, redirecting to detect_alerts.")
        return END

def build_graph(agent_obj):
    """Build the conversation graph"""
    builder = StateGraph(MessagesState)

    builder.add_node(agent_obj.list_tables)
    builder.add_node(agent_obj.call_get_schema)
    builder.add_node(agent_obj.get_schema_node, "get_schema")
    builder.add_node(agent_obj.generate_query)
    builder.add_node(agent_obj.check_query)
    builder.add_node(agent_obj.run_query_node, "run_query")
    builder.add_node(agent_obj.detect_alerts, "detect_alerts")

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "detect_alerts")
    builder.add_edge("detect_alerts", END)

    return builder.compile()

# Streamlit UI
def main():
    st.title("🔔 SQL Alert Detection Agent")
    st.markdown("---")

    # Initialize agent
    agent = initialize_agent()
    
    if agent is None:
        st.error("Failed to initialize the agent. Please check your configuration.")
        return

    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")
        email_recipient = st.text_input("Alert Email", value="sahil.singla@algo8.ai")
        st.session_state.email_recipient = email_recipient

        st.markdown("---")
        st.info("💡 Enter your SQL query to detect alerts in your database.")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Query")
        user_query = st.text_area(
            "Query",
            placeholder="e.g., based on recent pwani data, when the market share is high",
            height=100,
            label_visibility="collapsed"
        )
        
        submit_button = st.button("🔍 Analyze Query", type="primary", use_container_width=True)

    with col2:
        st.subheader("Status")
        status_container = st.container()

    # Process query
    if submit_button and user_query:
        # Reset status
        st.session_state.email_sent = False
        st.session_state.alert_detected = False
        st.session_state.sql_query = None
        st.session_state.query_results = None
        st.session_state.alert_message = None
        
        with st.spinner("🔄 Processing your query..."):
            try:
                user_query = " based on recent data" + user_query
                messages = [HumanMessage(content=user_query)]
                                
                                # Run the agent# Create placeholders for real-time updates
                sql_placeholder = st.empty()
                results_placeholder = st.empty()
                alert_placeholder = st.empty()

                # Run the agent
                for step in agent.stream({"messages": messages}, stream_mode="values"):
                    if step and "messages" in step and step["messages"]:
                        last_msg = step["messages"][-1]
                        if isinstance(last_msg, HumanMessage):
                            continue
                        
                        # Display SQL query as soon as it's generated
                        if st.session_state.sql_query and sql_placeholder:
                            with sql_placeholder.container():
                                st.markdown("#### 🔍 Generated SQL Query")
                                st.code(st.session_state.sql_query, language="sql")
                        
                       # Replace the entire Query Results rendering block with this:

                        # Display query results as soon as they're available
                        if st.session_state.query_results and results_placeholder:
                            with results_placeholder.container():
                                st.markdown("#### 📋 Query Results")

                                raw = st.session_state.query_results

                                try:
                                    # Case 1 — already a dict
                                    if isinstance(raw, dict):
                                        result_dict = raw

                                    # Case 2 — string → parse after cleaning datetime
                                    elif isinstance(raw, str):
                                        result_dict = safe_parse_result(raw)

                                    else:
                                        raise ValueError("Unsupported result format")

                                    # Convert to DataFrame
                                    df = convert_to_dataframe(result_dict)

                                    # Display only using st.table
                                    st.dataframe(df,use_container_width=True)

                                except Exception as e:
                                    st.error(f"Could not parse results: {e}")
                                    st.text(raw)

                st.markdown("---")
                st.markdown("### 📊 Results")

              
                # Display Alert Analysis
                if st.session_state.alert_message:
                    st.markdown("#### ⚠️ Alert Analysis")
                    
                    if st.session_state.alert_detected:
                        st.error(st.session_state.alert_message)
                    else:
                        st.success(st.session_state.alert_message)
                
                # Display status after processing
                with status_container:
                    if st.session_state.alert_detected:
                        st.error("⚠️ Alert Detected!")
                        if st.session_state.email_sent:
                            st.success("✅ Email sent successfully!")
                            st.info(f"📧 Notification sent to: {email_recipient}")
                        else:
                            st.warning("⚠️ Failed to send email notification")
                    else:
                        st.success("✅ No alerts detected")
                        st.info("ℹ️ All metrics are within normal range")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                logger.error(f"Error processing query: {e}")

    # Display message history
    # Removed query history section

if __name__ == "__main__":
    main()