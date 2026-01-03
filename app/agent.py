from app.tools import summarize

def run_agent(task, document):
    steps = []

    steps.append("Received task")
    
    steps.append("Analyzed document")
    
    summary = summarize(document)
    steps.append("Summarized document")

    result = {
        "task": task,
        "output": summary,
        "steps": steps
    }

    return result
