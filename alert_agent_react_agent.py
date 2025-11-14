import logging
import sys
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from my_Sql_connection import sql_connection
from Openi_ai import build_llm


# -------------------------------------------------------
# Logging setup
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("app_log.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Step 1: Connect to Database
# -------------------------------------------------------
try:
    engine = sql_connection()
    db = SQLDatabase(engine=engine)
    logger.info("✅ Database connection successful.")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    raise e

# -------------------------------------------------------
# Step 2: Initialize LLM
# -------------------------------------------------------
try:
    llm = build_llm()
    logger.info("✅ LLM initialization successful.")
except Exception as e:
    logger.error(f"❌ LLM initialization failed: {e}")
    raise e

# -------------------------------------------------------
# Step 3: Initialize SQL Toolkit
# -------------------------------------------------------
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
logger.info(f"✅ Toolkit initialized with {len(tools)} tools.")

# -------------------------------------------------------
# Step 4: Create ReAct Agent
# -------------------------------------------------------
SYSTEM_PROMPT = """
You are an intelligent SQL assistant.
You can:
- Retrieve database schema
- Generate SQL queries
- Execute queries using provided tools
- Detect anomalies or alerts in the results

Rules:
- NEVER change or delete data (SELECT only)
- Limit to 5 results unless specified
- After running a query, analyze results for anomalies:
  - unusually high/low values
  - missing data
  - unexpected patterns
If no issues, respond with "NO_ALERTS DETECTED".
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)

# -------------------------------------------------------
# Step 5: Optional helper for alert analysis (via LLM)

# -------------------------------------------------------
# Step 6: Interactive streaming version
# -------------------------------------------------------
def interactive_agent(agent):
    logger.info("🚀 Starting interactive SQL ReAct agent session.")

    # Example input (you can replace this with user input)
    question = "recent data which has high temperature in alerts_data"

    print(f"\n--- Processing Query ---\nUser: {question}\n")

    # Stream step-by-step reasoning and output
    try:
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            if step["messages"]:
                step["messages"][-1].pretty_print()

        # After streaming completes, extract the last response
      

    except Exception as e:
        logger.error(f"❌ Error during streaming: {e}")
        print(f"Error: {e}")

# -------------------------------------------------------
# Step 7: Run
# -------------------------------------------------------
if __name__ == "__main__":
    interactive_agent(agent)
