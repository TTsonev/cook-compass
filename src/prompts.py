intro_prompt = """
TASK:
- If the user asks for a recipe, provide one based ONLY on the provided CONTEXT. Give the Name, Ingredients, and Instructions
- If the CONTEXT is empty, say that you couldn't find a matching recipe.
- If the CONTEXT is empty, NEVER come up with a recipe that is not in the CONTEXT
- If the user asks you to modify a recipe that you have provided previously, you can do so based on your general knowledge.

CONTEXT: {context_content}

USER REQUEST: "{user_message}"
"""

rewrrite_query_prompt = """
            You are a helpful assistant improving search queries for a recipe database.
            Usually the current user request is clear enough, if not refine it based on the chat history.
            
            EXAMPLE 1:
                user asks for 'chicken soup' -> query is query -> you output 'chicken soup'
                          
            EXAMPLE 2:
                user asks for 'make it vegan' -> query is unclear -> you look at the chat history -> it mentions soup -> you output 'vegan soup'

            Based on the examples above, rewrite the user request into a specific, standalone search query that includes necessary context (like main ingredient)

            CURRENT USER QUERY: '{last_user_msg}'

            CONVERSATION HISTORY: {conversation_text}

            Output ONLY the rewritten search query. """

keywords_prompt = """Identify which of the following keywords are relevant to the user query: '{query}'.
Available keywords: {available_keywords}
Return ONLY a comma-separated list of the relevant keywords from the available list. If none match, return nothing."""
