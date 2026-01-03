from app.agent import run_agent

task = "Summarize this document"
document = "This is a sample document."

output = run_agent(task, document)
print(output)
