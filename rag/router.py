#Decides which path user wants - upload and query or global querying

class QueryRouter:
    def route(self, query, selected_pdf=None):
        q = query.lower()

        if "summarize" in q:
            return "summarize"

        if selected_pdf:
            return "upload_qa"

        return "global_qa"