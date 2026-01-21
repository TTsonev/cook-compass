intro_prompt = """
TASK:
- If the user asks for a recipe, provide one based ONLY on the provided CONTEXT. Give the Name, Ingredients, and Instructions
- If the CONTEXT is empty, say that you couldn't find a matching recipe.
- If the CONTEXT is empty, NEVER come up with a recipe that is not in the CONTEXT
- If the user asks you to modify a recipe that you have provided previously, you can do so based on your general knowledge.

CONTEXT: {context_content}

USER REQUEST: {user_message}
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

evaluation_prompt = """
                You are an impartial evaluation model acting as a judge for a recipe
                recommendation system called CookCompass.

                Your task is to evaluate the SYSTEM RESPONSE based on the USER QUERY
                and score it on the criteria listed below.

                Do NOT rewrite, improve, or explain the recipe.
                Only evaluate it.

                --------------------------------------------------
                USER QUERY:
                {user_query}

                --------------------------------------------------
                SYSTEM RESPONSE:
                {model_response}

                --------------------------------------------------
                EVALUATION CRITERIA:

                1. RELEVANCE (0-10):
                How well does the response satisfy the user's query and constraints?
                Consider:
                - Correct handling of dietary restrictions (e.g. vegetarian, allergies)
                - Use of requested ingredients
                - Avoidance of forbidden ingredients
                - Overall alignment with the intent of the query

                Score meaning:
                0 = Completely irrelevant or violates key constraints  
                5 = Partially relevant, but with notable issues  
                10 = Fully relevant and perfectly aligned with the query  

                2. HEALTHINESS (0-10):
                How healthy are the suggested recipe(s) in a general nutritional sense?
                Consider:
                - Balance of ingredients (e.g. vegetables, fats, protein)
                - Excessive use of sugar, salt, or ultra-processed ingredients
                - Suitability for everyday consumption (not medical advice)

                Score meaning:
                0 = Extremely unhealthy  
                5 = Moderately healthy  
                10 = Very healthy and well-balanced  

                3. TASTE (0-10):
                How appealing and plausible are the recipe(s) from a culinary perspective?
                Consider:
                - Ingredient compatibility
                - Flavor balance
                - Whether the recipe makes sense to a typical cook

                Score meaning:
                0 = Unappetizing or nonsensical  
                5 = Acceptable but unexciting  
                10 = Highly appealing and well-composed  

                --------------------------------------------------
                OUTPUT FORMAT (STRICT JSON):

                {{
                "relevance": <integer 0-10>,
                "healthiness": <integer 0-10>,
                "taste": <integer 0-10>
                }}
                """