from mcp_server.timesheet_mcp import mcp

if __name__ == "__main__":
    print("🔌 Timesheet MCP server → http://localhost:8895/sse")
    mcp.run(transport="sse", port=8895)