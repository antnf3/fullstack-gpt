import streamlit as st
import os
import requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import SystemMessage


llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-1106")

alpha_vantage_api = os.environ.get("ALPHA_VANTAGE_API_KEY")


# Pydantic을 이용하여  argument의 유형을 정의한다.
class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


# BaseTool을 이용하여 Tool을 만든다.
class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.
    Example query: Stock Makret Symbol for Apple Company
    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = (
        StockMarketSymbolSearchToolArgsSchema
    )

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


class CompanyOverviewToolArgsSchema(BaseModel):
    symbol: str = Field(description="Stock symbol of the company.Example: AAPL,TSLA")


class CompanyOverviewTool(BaseTool):
    name = "CompanyOverviewTool"
    description = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """

    args_schema: Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api}"
        )
        return r.json()


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatementTool"
    description = """
    Use this to get the income statement of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api}"
        )
        return r.json()["annualReports"]


class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformanceTool"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api}"
        )
        return list(r.json()["Weekly Time Series"].items())[:200]
        # return r.json()


agent = initialize_agent(
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a hedge fund manager.
            
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
            
            Consider the performance of a stock, the company overview and the income statement.
            
            Be assertive in your judgement and recommend the stock or advise the user against it.
        """
        )
    },
)


st.set_page_config(page_title="InvestGPT", page_icon="$")

st.markdown(
    """
    # InvestorGPT

    Welcome to Investor GPT

    Write down the name of a company and our Agent will do the reseach for you.
    """
)

company = st.text_input(label="관심있는 회사명을 입력하세요.")

if company:
    result = agent.invoke(company)
    st.write(result["output"].replace("$", "\$"))
