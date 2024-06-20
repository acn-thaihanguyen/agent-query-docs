import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from llama_index.core.agent import FunctionCallingAgentWorker, ReActAgent
from llama_index.core.memory import (ChatMemoryBuffer, SimpleComposableMemory,
                                     VectorMemory)
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI

from src.prompts.prompts import context, instruction_str, new_prompt
from src.tools.pdf import create_pdf_engine

# Load environment variables from a .env file
load_dotenv()

# Path to the memo file
NOTE_FILE = os.path.join(os.path.dirname(__file__), "output", "memo.txt")


def save_note(note):
    """
    Save a note to the memo file with a formatted header.
    """
    header = f"# {'Memo':^76} #\n"
    footer = f"# {'-' * 76} #\n"

    # Create memo file if it doesn't exist
    if not os.path.exists(NOTE_FILE):
        with open(NOTE_FILE, "w") as f:
            f.write(header)
            f.write(note + "\n")
            f.write(footer)

    # Append the note to the memo file
    with open(NOTE_FILE, "a") as f:
        f.write(header)
        f.writelines(note + "\n")
        f.write(footer)

    return "Successfully saved note to memo file!"


class NoteEngine:

    @staticmethod
    def create():
        """
        Create a function tool to save notes.

        Returns:
            FunctionTool: A tool for saving notes.
        """
        return FunctionTool.from_defaults(
            name="NoteEngine",
            fn=save_note,
            description="Save a note to a memo file",
        )


class PandasQueryEngineFactory:

    @staticmethod
    def create(csv_path, instruction_str, new_prompt):
        """
        Create a query engine for a CSV file.

        Args:
            csv_path (str): Path to the CSV file.
            instruction_str (str): Instructions for the query engine.
            new_prompt (str): Prompt for the query engine.

        Returns:
            PandasQueryEngine: The query engine for the CSV file.
        """
        df = pd.read_csv(csv_path)
        query_engine = PandasQueryEngine(
            df, verbose=True, instruction_str=instruction_str
        )
        query_engine.update_prompts({"pandas_prompt": new_prompt})
        return query_engine


class AgentApp:

    def __init__(self, csv_path, pdf_path, agent_type="FunctionCallingAgentWorker"):
        self.csv_path = csv_path
        self.pdf_path = pdf_path
        self.agent_type = agent_type
        self.tools = []
        self.llm = OpenAI(model=os.environ["OPENAI_MODEL"], temperature=0)

    def setup_tools(self):

        # Create note engine
        note_engine = NoteEngine.create()
        self.tools.append(note_engine)

        # Create query engines for CSV files
        query_engine = PandasQueryEngineFactory.create(
            self.csv_path, instruction_str, new_prompt
        )
        self.tools.append(
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="CsvQueryEngine",
                    description="Query the dataframe!",
                ),
            )
        )

        # Create query engines for PDF files
        pdf_engine = create_pdf_engine(self.pdf_path)
        self.tools.append(
            QueryEngineTool(
                query_engine=pdf_engine,
                metadata=ToolMetadata(
                    name="PdfQueryEngine",
                    description="Query the PDF data",
                ),
            )
        )

    def run_agent(self):
        # Create a vector memory for the agent
        vector_memory = VectorMemory.from_defaults(
            vector_store=None,  # Leave as None to use default in-memory vector store
            embed_model=OpenAIEmbedding(),
            retriever_kwargs={"similarity_top_k": 1},
        )

        chat_memory_buffer = ChatMemoryBuffer.from_defaults()
        composable_memory = SimpleComposableMemory.from_defaults(
            primary_memory=chat_memory_buffer,
            # secondary_memory_sources=[vector_memory],
        )

        if self.agent_type == "ReActAgent":
            agent = ReActAgent.from_tools(
                self.tools,
                llm=self.llm,
                verbose=True,
                context=context,
                memory=composable_memory,
            )

        elif self.agent_type == "FunctionCallingAgentWorker":
            agent_worker = FunctionCallingAgentWorker.from_tools(
                self.tools,
                llm=self.llm,
                verbose=True,
                system_prompt=context,
            )
            agent = agent_worker.as_agent(
                memory=composable_memory,
                verbose=True,
            )
        else:
            raise ValueError("Invalid agent type")

        while (prompt := input("Enter a prompt (q to quit): ")) != "q":
            result = agent.chat(prompt)
            print(result)


def main(csv_path, pdf_path, agent_type="ReActAgent"):
    """
    Main function to create and run the agent with specified tools.

    Args:
        csv_path (list): List of CSV file paths.
        pdf_path (list): List of PDF file paths.
        agent_type (str): Type of agent to use (default is "ReActAgent").
    """
    app = AgentApp(csv_path, pdf_path, agent_type)
    app.setup_tools()
    app.run_agent()


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Query CSV and PDF files using ReActAgent."
    )
    parser.add_argument("--csv", help="List of CSV files to query.", required=False)
    parser.add_argument("--pdf", help="List of PDF files to query.", required=False)
    parser.add_argument("--agent", help="Agent number (1:ReActAgent or 2:FunctionCallingAgentWorker)", required=False)
    args = parser.parse_args()

    # Run the main function with the specified CSV and PDF files
    csv_path = args.csv if args.csv else "./data/LLMs.csv"
    pdf_path = args.pdf if args.pdf else "./data/principled_instructions.pdf"
    agent_type = int(args.agent)
    
    if agent_type == 1:
        agent_type = "ReActAgent"
    elif agent_type == 2:
        agent_type = "FunctionCallingAgentWorker"
    else:
        raise ValueError("Invalid agent type")

    # Run the main function
    main(csv_path, pdf_path, agent_type)
