import os
from dotenv import load_dotenv
from google.adk.dev import app

from finance_agent.agent import finance_agent

load_dotenv()

# Register your agent with ADK Dev server
app.register_agent(finance_agent)