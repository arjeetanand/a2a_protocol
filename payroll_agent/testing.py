# test_payroll_tools.py
import os
import asyncio
from agent import get_employee_hours_via_mcp

print("Testing MCP Tool...")
result = get_employee_hours_via_mcp("arjeet")
print(result)



from agent import calculate_gross_pay

print("Testing Gross Pay Tool...")
# print(calculate_gross_pay("arjeet", 50.5))


from agent import generate_payslip_pdf

gross_json = calculate_gross_pay("arjeet", 40.5)

project_breakdown_json = '{"ABC": 10.2}'

print("Testing PDF generation...")
print(generate_payslip_pdf("arjeet", gross_json, project_breakdown_json))



import asyncio
from agent import build_payroll_agent

# async def test():
#     agent = build_payroll_agent()
#     result = await agent.run("calculate gross pay for arjeet")
#     print(result.text)

# asyncio.run(test())