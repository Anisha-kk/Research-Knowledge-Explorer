# rag/summarizer.py

class Summarizer:
    def __init__(self, llm):
        self.llm = llm

    def summarize(self, docs):
        text = "\n\n".join(d.get("text", "") for d in docs if d.get("text"))
        
        prompt = f"""
        You are a scientific assistant.

        Summarize ONLY the content provided below.
        Do NOT add external knowledge.
        Do NOT invent concepts.

        If something is not present in the text, ignore it.
        {text}
        """

        return self.llm.generate(prompt)