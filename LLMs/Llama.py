from llama_index.llms.groq import Groq

def getLlamaResponse(GROQ,model,prompt):
  llm_groq = Groq(model=model, api_key=GROQ)
  sequences = llm_groq.complete(prompt = prompt)
  return sequences.text