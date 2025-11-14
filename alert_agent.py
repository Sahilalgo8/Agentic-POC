import logging
import sys
from langchain_community.utilities import SQLDatabase
from langchain.messages import AIMessage, HumanMessage
from send_mail import send_email
# from my_Sql_connection import sql_connection
from sqlalchemy import text

from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from my_Sql_connection import sql_connection
from Openi_ai import build_llm
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from langgraph.graph import END, START,StateGraph,MessagesState
from typing import Literal
from langchain.tools import tool
import ast

logger=logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("app_log.txt", encoding='utf-8'),  # ✅ file supports UTF-8
        logging.StreamHandler(sys.stdout)                      # ✅ console handler
    ]
)
logger = logging.getLogger(__name__)

try:
    engine = sql_connection()
    logger.info("Database connection successful.")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    raise e

# Step 1: Initialize the language model (LLM) using your build_llm function with error handling
try:
    llm = build_llm()
    logger.info("LLM initialization successful.")
except Exception as e:
    logger.error(f"LLM initialization failed: {e}")
    raise e

class alert_agent:
    def __init__(self, engine, llm):
        # Create LangChain SQLDatabase utility
        self.db = SQLDatabase(engine=engine)
        logger.info("SQLDatabase object created.")

        # Initialize LLM and SQL toolkit
        self.llm = llm
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        logger.info(f"Toolkit initialized with {len(self.tools)} tools.")

        # Extract key tools
        self.get_schema_tool = next(tool for tool in self.tools if tool.name == "sql_db_schema")
        self.run_query_tool = next(tool for tool in self.tools if tool.name == "sql_db_query")
        self.list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")
        logger.info("Key tools mapped.")

        # Wrap tools as nodes
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

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""

        system_message = {"role": "system", "content": prompt}
        llm_with_tools = self.llm.bind_tools([self.run_query_structured_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])

        # printing the query generated

        # query_tool_call = response.tool_calls[0]
        # query_tool_call=query_tool_call["args"]["query"]
        # logger.info(f"Query generated --> {query_tool_call} ")
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
        query_tool_call=query_tool_call["args"]["query"]
        logger.info(f"Query generated --> {query_tool_call} ")
        response.id = state["messages"][-1].id
        return {"messages": [response]}
    def detect_alerts(self, state: MessagesState):
        """
        Analyze query results and detect alerts using LLM.
        If alerts are found, send an email notification.
        """
        logger.info("Executing detect_alerts.")
        
        # Extract the user's original query from the first message
        original_query = None
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                original_query = message.content
                break
        
        # Get the query results
        last_message = state["messages"][-1]
        query_result = last_message.content

        prompt = f"""
        You are an alert detection system. Analyze the following query results and identify any alerts or anomalies.
        
        **User's Original Query:** {original_query}
                
        **Query Results:**
        {query_result}
        
        Look for:
        - Unusual values or patterns
        - Threshold violations (very high/low values)
        - Data inconsistencies
        - Any values that seem out of the ordinary
        - Values that deviate from expected business logic
        
        If no alerts are found, respond with "NO_ALERTS DETECTED".
        """

        system_message = {"role": "system", "content": prompt}

        response = self.llm.invoke([system_message])
        analysis_text = response.content.strip()

        # ----------------------------
        # EMAIL NOTIFICATION LOGIC
        # ----------------------------
        if "NO_ALERTS DETECTED" not in analysis_text:
            try:
                alert_subject = "Alert Detected in System"
                alert_body = f"{analysis_text}"
                
                send_email(
                    alert_subject,
                    alert_body,
                    "sahil.singla@algo8.ai"
                )
                logger.info("Alert email sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send alert email: {e}")

        # ----------------------------
        # Return alert analysis
        # ----------------------------
        alert_message = AIMessage(content=f"Alert Analysis:\n{analysis_text}")
        return {"messages": [alert_message]}

# -------------------------------------------------------
# Step 8: Instantiate the agent class
# -------------------------------------------------------
agent_obj = alert_agent(engine=engine, llm=llm)

# -------------------------------------------------------
# Step 9: Control flow function
# -------------------------------------------------------
def should_continue(state: MessagesState) -> Literal["check_query", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("Tool calls detected, continuing to check_query.")
        return "check_query"
    else:
        logger.info("No tool calls detected, redirecting to detect_alerts.")
        return END

# -------------------------------------------------------
# Step 10: Build the conversation graph
# -------------------------------------------------------
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

agent = builder.compile()

# -------------------------------------------------------
# Step 11: Interactive loop with user query list
# -------------------------------------------------------
def interactive_agent(agent):
    logger.info("Starting interactive session.")
    # while True:
        # user_input = input("\nEnter your query (or 'exit' to quit):\nYou: ").strip()
        # if user_input.lower() in ("exit", "quit"):
        #     logger.info("Exiting. Goodbye!")
        #     break
    user_input = "based on recent pwani data, when the white space  higher in last three month, alert me"

    logger.info(f"Processing query: {user_input[:50]}...")
    print(f"\n--- Processing Query ---")
    
    messages = [HumanMessage(content=user_input)]
    for step in agent.stream({"messages": messages}, stream_mode="values"):
        if step and "messages" in step and step["messages"]:
            last_msg = step["messages"][-1]
            if isinstance(last_msg, HumanMessage):
                continue
            print(f"\nAgent: {last_msg.content}")

if __name__ == "__main__":
    interactive_agent(agent)