# LLM reasoning

class QAChain:
    def __init__(self, llm):
        self.llm = llm

    def answer(self, query, docs):
        context = "\n\n".join(d.get("text", "") for d in docs if d.get("text"))
    
        prompt = f"""
        You are a strict research assistant.
        - If the answer is not in context, say "Not found in provided documents".
        - Do NOT guess citations or publication details.
        If you mention a paper, cite ONLY from context.
        Do not invent venues or years.
        Context:
        {context}
        Question:
        {query}
        in simple terms.
        Avoid citations unless necessary.
        """

        return self.llm.generate(prompt)