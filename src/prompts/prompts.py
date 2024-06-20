from llama_index.core import PromptTemplate

instruction_str = """\
    1. Convert the query into executable Python code using the Pandas library.
    2. Ensure the final line of code is a Python expression that can be executed using the `eval()` function.
    3. The code should solve the query accurately.
    4. Only print the expression without quoting it."""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    Here is a preview of the dataframe:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

context = """\
    Purpose: The primary role of this agent is to assist users by providing \
    accurate information about the content of the CSV and PDF files."""
