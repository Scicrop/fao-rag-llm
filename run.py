import os
import sys
from decouple import config
from llama_index.core.agent import (
    CustomSimpleAgentWorker,
    Task,
    AgentChatResponse,
)
from typing import Dict, Any, List, Tuple, Optional
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core import ChatPromptTemplate, PromptTemplate
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.tools import QueryEngineTool
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
    Float,
    Date
)
from llama_index.core import SQLDatabase
import pandas as pd
from datetime import datetime
from sqlalchemy import insert
from llama_index.core.agent import AgentRunner
from llama_index.llms.openai import OpenAI

DEFAULT_PROMPT_STR = """
Given previous question/response pairs, please determine if an error has occurred in the response, and suggest \
    a modified question that will not trigger the error.

Examples of modified questions:
- The question itself is modified to elicit a non-erroneous response
- The question is augmented with context that will help the downstream system better answer the question.
- The question is augmented with examples of negative responses, or other negative questions.

An error means that either an exception has triggered, or the response is completely irrelevant to the question.

Please return the evaluation of the response in the following JSON format.

"""


def line_wrap(text, max_len=200):
    return "\n".join(text[i:i + max_len] for i in range(0, len(text), max_len))


def get_chat_prompt_template(
        system_prompt: str, current_reasoning: Tuple[str, str]
) -> ChatPromptTemplate:
    system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
    messages = [system_msg]
    for raw_msg in current_reasoning:
        if raw_msg[0] == "user":
            messages.append(
                ChatMessage(role=MessageRole.USER, content=raw_msg[1])
            )
        else:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=raw_msg[1])
            )
    return ChatPromptTemplate(message_templates=messages)


class ResponseEval(BaseModel):
    """Evaluation of whether the response has an error."""

    has_error: bool = Field(
        ..., description="Whether the response has an error."
    )
    new_question: str = Field(..., description="The suggested new question.")
    explanation: str = Field(
        ...,
        description=(
            "The explanation for the error as well as for the new question."
            "Can include the direct stack trace as well."
        ),
    )


class RetryAgentWorker(CustomSimpleAgentWorker):
    prompt_str: str = Field(default=DEFAULT_PROMPT_STR)
    max_iterations: int = Field(default=1)

    _router_query_engine: RouterQueryEngine = PrivateAttr()

    def __init__(self, tools: List[BaseTool], **kwargs: Any) -> None:
        """Init params."""
        # validate that all tools are query engine tools
        for tool in tools:
            if not isinstance(tool, QueryEngineTool):
                raise ValueError(
                    f"Tool {tool.metadata.name} is not a query engine tool."
                )
        self._router_query_engine = RouterQueryEngine(
            selector=PydanticSingleSelector.from_defaults(),
            query_engine_tools=tools,
            verbose=kwargs.get("verbose", False),
        )
        super().__init__(
            tools=tools,
            **kwargs,
        )

    def _initialize_state(self, task: Task, **kwargs: Any) -> Dict[str, Any]:
        """Initialize state."""
        return {"count": 0, "current_reasoning": []}

    def _run_step(
            self, state: Dict[str, Any], task: Task, input: Optional[str] = None
    ) -> Tuple[AgentChatResponse, bool]:

        if "new_input" not in state:
            new_input = task.input
        else:
            new_input = state["new_input"]

        # first run router query engine
        response = self._router_query_engine.query(new_input)

        # append to current reasoning
        state["current_reasoning"].extend(
            [("user", new_input), ("assistant", str(response))]
        )

        chat_prompt_tmpl = get_chat_prompt_template(
            self.prompt_str, state["current_reasoning"]
        )
        llm_program = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(output_cls=ResponseEval),
            prompt=chat_prompt_tmpl,
            llm=self.llm,
        )
        # run program, look at the result
        response_eval = llm_program(
            query_str=new_input, response_str=str(response)
        )
        if not response_eval.has_error:
            is_done = True
        else:
            is_done = False
        state["new_input"] = response_eval.new_question

        #response = line_wrap(str(response))

        if self.verbose:
            print(f"- Question: {new_input}")
            print(f"- Response: {response}")
            print(f"- Response eval: {response_eval.dict()}")

        # return response
        return AgentChatResponse(response=str(response)), True

    def _finalize_task(self, state: Dict[str, Any], **kwargs) -> None:
        """Finalize task."""
        # nothing to finalize here
        # this is usually if you want to modify any sort of
        # internal state beyond what is set in `_initialize_state`
        pass


def prepare(data_sources):
    engine = create_engine("sqlite:///:memory:", future=True)
    metadata_obj = MetaData()
    file_name = "Food_price_indices_data_mar24.csv"
    df = pd.read_csv("./data/" + file_name)
    print("Processing data from: " + file_name)
    sql_tools = []
    vector_tools = []
    for ds in data_sources:
        data_array = []
        for index, row in df.iterrows():
            date = str(row['Date'])
            if date is None or date == 'nan':
                continue
            data = row[ds['type']]
            data_obj = datetime.strptime(str(date), "%Y-%m")
            data_array.append({"Date": data_obj, "Price": data})
        print("Processing data of: " + ds["type"])

        table_name = ds['table']
        stats_table = Table(
            table_name,
            metadata_obj,
            Column("Date", Date, primary_key=True),
            Column("Price", Float),
        )

        metadata_obj.create_all(engine)

        for row in data_array:
            stmt = insert(stats_table).values(**row)
            with engine.begin() as connection:
                cursor = connection.execute(stmt)

        from llama_index.core.query_engine import NLSQLTableQueryEngine

        sql_database = SQLDatabase(engine, include_tables=[table_name])
        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, tables=[table_name], verbose=True
        )

        description = (
                "Useful for translating a natural language query into a SQL query over"
                " a table containing: " + table_name + ", containing the date/price of"
                                                       " of " + ds['type']
        )

        sql_tool = QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            description=description,
        )

        sql_tools.append(sql_tool)

        from llama_index.core import (
            SimpleDirectoryReader,
            VectorStoreIndex,
            download_loader,
            RAKEKeywordTableIndex,
        )

        reader = SimpleDirectoryReader(input_files=["./data/" + ds['file_pdf']])
        data = reader.load_data()
        vector_index = VectorStoreIndex.from_documents(data)
        vector_query_engine = vector_index.as_query_engine(streaming=True, similarity_top_k=3)
        embedding = "Useful for answering semantic questions about " + ds["type"]
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=embedding,
        )
        vector_tools.append(vector_tool)

    llm = OpenAI(model="gpt-4")
    callback_manager = llm.callback_manager

    query_engine_tools = sql_tools + vector_tools
    agent_worker = RetryAgentWorker.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=False,
        callback_manager=callback_manager,
    )
    agent = AgentRunner(agent_worker, callback_manager=callback_manager)

    must_continue = True
    while (must_continue):
        question = input("> ")
        if question == "quit":
            must_continue = False
            break
        response = agent.chat(question)
        print(str(response))


def main(argv):
    OPENAI_KEY = config('OPENAI_KEY')
    if OPENAI_KEY is None or OPENAI_KEY == '':
        os.environ['OPENAI_API_KEY'] = input("Please insert your OPENAI key: ")
    else:
        os.environ['OPENAI_API_KEY'] = OPENAI_KEY
    data_sources = []
    data_sources.append({"type": "Oils", "table": "oils_prices", "file_pdf": "cd0156en.pdf"})
    data_sources.append({"type": "Meat", "table": "meat_prices", "file_pdf": "cc9074en.pdf"})
    data_sources.append({"type": "Dairy", "table": "dairy_prices", "file_pdf": "cc9105en.pdf"})
    data_sources.append({"type": "Cereals", "table": "cereals_prices", "file_pdf": "Cereals.pdf"})
    data_sources.append({"type": "Sugar", "table": "sugar_prices", "file_pdf": "Sugar.pdf"})
    prepare(data_sources)


if __name__ == "__main__":
    main(sys.argv)
