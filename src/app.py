import streamlit as st
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from colorama import Fore, Style, init
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import Image
import tempfile
import yfinance as yf
import os
# Import local modules
from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.warren_buffett import warren_buffett_agent
from agents.valuation import valuation_agent
from graph.state import AgentState
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from utils.visualize import save_graph_as_png
from llm.models import LLM_ORDER, get_model_info
from langgraph.graph import END, StateGraph
import opik
from opik.integrations.langchain import OpikTracer

# Load environment variables
load_dotenv()
init(autoreset=True)

try:
    opik_url = os.environ["OPIK_URL_OVERRIDE"]
except:
    opik_url = "https://www.comet.com/opik/api/"
    os.environ["OPIK_URL_OVERRIDE"] = opik_url

try:
    opik_workspace = os.environ["OPIK_WORKSPACE"]
except:
    opik_workspace = "harikrishnad1997"
    os.environ["OPIK_WORKSPACE"] = opik_workspace

try:
    opik_project_name = os.environ["OPIK_PROJECT_NAME"]
except:
    opik_project_name = "AI_hedge_fund_tesing"
    os.environ["OPIK_PROJECT_NAME"] = opik_project_name

# Set page configuration
st.set_page_config(
    page_title="Hari's AI Hedge Fund Trading System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        st.error(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        st.error(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None

def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state

def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow

def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
    progress_bar=None
):
    try:
        # Create a new workflow if analysts are customized
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()

            tracer = OpikTracer(graph=agent.get_graph(xray=True))
        else:
            agent = create_workflow(selected_analysts=["charlie_munger"]).compile()
            tracer = OpikTracer(graph=agent.get_graph(xray=True))

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
            config={'callbacks':[tracer]}
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return {"decisions": None, "analyst_signals": {}}

# Streamlit app components
def display_header():
    st.title("Hari's AI Hedge Fund Trading System")
    # st.image("src/assets/about.jpg", use_container_width=True)
    st.markdown("""
    This application uses AI agents to analyze stocks and make trading decisions.
    Select your preferences and run the simulation to see what the AI recommends.
    """)

# @st.cache_data
# def get_tickers():
#     sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
#     nasdaq = pd.read_csv("https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv")['Symbol'].tolist()
#     dow = ["AAPL", "MSFT", "AMZN", "JNJ", "V", "PG", "UNH", "NVDA", "JPM", "HD"]  # Dow Jones doesn't change often

#     # Combine tickers and remove duplicates
#     return sorted(set(sp500 + nasdaq + dow))


def display_sidebar_settings():
    # st.sidebar.image("[![Logo](assets/Hari.jpeg)](https://www.linkedin.com/in/hdev1997/)", unsafe_allow_html=True)
    st.sidebar.title("Harikrishna Dev")
    st.sidebar.image("src/assets/Hari.jpeg", use_container_width=True)
    # st.sidebar.markdown("""<a href="https://www.linkedin.com/in/hdev1997/">Connect with me on LinkedIn</a>""", unsafe_allow_html=True)
    st.sidebar.markdown("""
<div style="display: flex; gap: 10px;">
    <a href="https://www.linkedin.com/in/hdev1997/" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="30">
    </a>
    <a href="https://github.com/hdev1997" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" width="30">
    </a>
    <a href="mailto:harikrish0607@gmail.com" target="_blank">
        <img src="https://img.icons8.com/ios-glyphs/30/000000/new-post.png" alt="Email" width="30">
    </a>
    <a href="https://topmate.io/harikrishnad" target="_blank" rel="noopener noreferrer">
        <img src="https://img.icons8.com/ios-filled/50/000000/phone.png" alt="Topmate" width="30">
    </a>
    <a href="https://medium.com/@harikrishnad1997" target="_blank" rel="noopener noreferrer">
        <img src="https://cdn-icons-png.flaticon.com/512/5968/5968885.png" alt="Medium" width="30">
    </a>
</div>
""", unsafe_allow_html=True)

    st.sidebar.header("Configuration")

    password = st.sidebar.text_input("Password", type="password")
    if password != os.getenv("PASSWORD"):
        st.error("Enter the right password: HARIHATESAI all L0w3RCA5E")
        st.stop()
    # Portfolio settings
    st.sidebar.subheader("Portfolio Settings")
    initial_cash = st.sidebar.number_input("Initial Cash ($)", value=100000.0, step=10000.0)
    margin_requirement = st.sidebar.number_input("Margin Requirement ($)", value=0.0, step=1000.0)
    # sp500_tickers = [ticker for ticker in yf.Ticker("^GSPC").history(period="1d").index]
    # nasdaq_tickers = [ticker for ticker in yf.Ticker("^IXIC").history(period="1d").index]
    # dow_tickers = [ticker for ticker in yf.Ticker("^DJI").history(period="1d").index]
    # # Ticker settings
    # tickers = sorted(set(sp500_tickers + nasdaq_tickers + dow_tickers))
    # st.write("Available Stock Tickers:")
    # st.write(tickers)
    ticker_input = st.sidebar.text_input("Stock Tickers (comma-separated)", "AAPL, MSFT, AMZN")
#     ticker_input = st.sidebar.multiselect(
#     "Select Stock Tickers", 
#     options=tickers, 
#     default=["AAPL", "MSFT", "GOOGL"]
# )
    tickers = [ticker.strip() for ticker in ticker_input.split(",")]
    
    # Date settings
    st.sidebar.subheader("Date Range")
    end_date = st.sidebar.date_input("End Date", datetime.now())
    start_date = st.sidebar.date_input("Start Date", (end_date - relativedelta(months=12)))
    
    # Additional settings
    st.sidebar.subheader("Additional Settings")
    show_reasoning = st.sidebar.checkbox("Show Agent Reasoning", value=True)
    show_agent_graph = st.sidebar.checkbox("Show Agent Graph", value=False)
    
    return {
        "initial_cash": initial_cash,
        "margin_requirement": margin_requirement,
        "tickers": tickers,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "show_reasoning": show_reasoning,
        "show_agent_graph": show_agent_graph
    }

def select_analysts():
    st.subheader("Select AI Analysts")
    
    # Convert ANALYST_ORDER into a dictionary for Streamlit multiselect
    analyst_options = {value: display for display, value in ANALYST_ORDER}
    
    # Default to all analysts selected
    default_analysts = list(analyst_options.keys())[0:3]
    
    selected_analysts = st.multiselect(
        "Choose which analysts to include in your simulation:",
        options=list(analyst_options.keys()),
        default=default_analysts,
        format_func=lambda x: analyst_options[x]
    )
    
    if not selected_analysts:
        st.warning("You must select at least one analyst.")
    
    return selected_analysts

def select_llm_model():
    st.subheader("Select LLM Model")
    
    # Convert LLM_ORDER into a format for Streamlit selectbox
    model_options = {value: display for display, value, _ in LLM_ORDER}
    
    model_choice = st.selectbox(
        "Choose which LLM model to use:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index = 7,
        # default = "gemini-2.0-flash",
    )
    
    model_info = get_model_info(model_choice)
    if model_info:
        model_provider = model_info.provider.value
        st.info(f"Selected {model_provider} model: {model_choice}")
    else:
        model_provider = "Unknown"
        st.info(f"Selected model: {model_choice}")
    
    return model_choice, model_provider

# def display_results(result):
    if not result or not result.get("decisions"):
        st.error("No valid results were returned.")
        return
    
    decisions = result["decisions"]
    analyst_signals = result["analyst_signals"]
    
    st.header("Trading Decisions")
    
    # Display overall portfolio information
    if "portfolio_summary" in decisions:
        st.subheader("Portfolio Summary")
        summary = decisions["portfolio_summary"]
        
        # Create a DataFrame for the summary
        summary_df = pd.DataFrame([{
            "Initial Cash": f"${summary.get('initial_cash', 0):,.2f}",
            "Current Cash": f"${summary.get('current_cash', 0):,.2f}",
            "Total Portfolio Value": f"${summary.get('total_portfolio_value', 0):,.2f}",
            "Realized Gains": f"${summary.get('total_realized_gains', 0):,.2f}",
            "Unrealized Gains": f"${summary.get('total_unrealized_gains', 0):,.2f}"
        }])
        
        st.dataframe(summary_df, use_container_width=True)
    
    # Display position information
    if "positions" in decisions:
        st.subheader("Current Positions")
        
        positions = decisions["positions"]
        positions_data = []
        
        for ticker, position in positions.items():
            positions_data.append({
                "Ticker": ticker,
                "Long Shares": position.get("long", 0),
                "Short Shares": position.get("short", 0),
                "Long Cost Basis": f"${position.get('long_cost_basis', 0):,.2f}",
                "Short Cost Basis": f"${position.get('short_cost_basis', 0):,.2f}",
                "Current Value": f"${position.get('current_value', 0):,.2f}",
                "Realized Gain/Loss": f"${position.get('realized_gains', 0):,.2f}",
                "Unrealized Gain/Loss": f"${position.get('unrealized_gains', 0):,.2f}"
            })
        
        positions_df = pd.DataFrame(positions_data)
        st.dataframe(positions_df, use_container_width=True)
    
    # Display trading decisions
    if "trading_decisions" in decisions:
        st.subheader("Trading Decisions")
        
        trading_decisions = decisions["trading_decisions"]
        decisions_data = []
        
        for ticker, decision in trading_decisions.items():
            decisions_data.append({
                "Ticker": ticker,
                "Action": decision.get("action", "HOLD"),
                "Shares": decision.get("shares", 0),
                "Price": f"${decision.get('price', 0):,.2f}",
                "Reasoning": decision.get("reasoning", "")
            })
        
        decisions_df = pd.DataFrame(decisions_data)
        st.dataframe(decisions_df, use_container_width=True)
    
    # Display analyst signals if show_reasoning is enabled
    if analyst_signals and st.checkbox("Show Analyst Signals", value=False):
        st.subheader("Analyst Signals")
        
        for analyst, signals in analyst_signals.items():
            with st.expander(f"{analyst.replace('_', ' ').title()} Analysis"):
                for ticker, signal in signals.items():
                    st.write(f"**{ticker}**: {signal}")

def display_results(result):
    if not result or not result.get("decisions"):
        st.error("No valid results were returned.")
        return
    
    decisions = result["decisions"]
    analyst_signals = result["analyst_signals"]
    
    st.header("Trading Decisions")
    
    # Display trading decisions
    st.subheader("Trading Decisions")
    
    decisions_data = []
    for ticker, decision in decisions.items():
        decisions_data.append({
            "Ticker": ticker,
            "Action": decision.get("action", "HOLD").upper(),
            "Quantity": decision.get("quantity", 0),
            "Confidence": f"{decision.get('confidence', 0)}%",
            "Reasoning": decision.get("reasoning", "")
        })
    
    decisions_df = pd.DataFrame(decisions_data)
    st.dataframe(decisions_df, use_container_width=True)
    
    # Display analyst signals
    if analyst_signals:
        st.subheader("Analyst Signals")
        
        for analyst, signals in analyst_signals.items():
            with st.expander(f"{analyst.replace('_', ' ').title()} Analysis"):
                for ticker, signal_data in signals.items():
                    st.markdown(f"### {ticker}")
                    
                    # Different handling for risk_management_agent
                    if analyst == "risk_management_agent":
                        st.markdown(f"**Remaining Position Limit:** ${signal_data.get('remaining_position_limit', 0):,.2f}")
                        st.markdown(f"**Current Price:** ${signal_data.get('current_price', 0):,.2f}")
                        
                        if isinstance(signal_data.get('reasoning'), dict):
                            reasoning = signal_data['reasoning']
                            st.markdown("**Risk Assessment:**")
                            st.dataframe(reasoning, use_container_width=True)
                            # st.json(reasoning)
                    else:
                        st.markdown(f"**Signal:** {signal_data.get('signal', 'N/A')}")
                        st.markdown(f"**Confidence:** {signal_data.get('confidence', 0)}%")
                        
                        if signal_data.get('reasoning'):
                            st.markdown("**Reasoning:**")
                            st.markdown(signal_data['reasoning'])
                    
                    st.markdown("---")

def main():
    display_header()
    settings = display_sidebar_settings()
    
    # Create columns for the selection areas
    col1, col2 = st.columns(2)
    
    with col1:
        selected_analysts = select_analysts()
    
    with col2:
        model_choice, model_provider = select_llm_model()
    
    # Initialize portfolio with cash amount and stock positions
    portfolio = {
        "cash": settings["initial_cash"],
        "margin_requirement": settings["margin_requirement"],
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
            } for ticker in settings["tickers"]
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            } for ticker in settings["tickers"]
        }
    }
    
    # Run button and visualization
    if st.button("Run Hedge Fund Simulation", type="primary"):
        if not selected_analysts:
            st.error("Please select at least one analyst.")
            return
        
        with st.spinner("Running simulation..."):
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Display the agent graph if requested
            if settings["show_agent_graph"]:
                workflow = create_workflow(selected_analysts)
                app = workflow.compile()
                
                graph_image = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

                # Write the image to a temporary file
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                #     tmpfile.write(graph_image.data)
                #     tmpfile_path = tmpfile.name

                # graph_name = "_".join(selected_analysts) + "_graph.png"
                # graph_path = save_graph_as_png(app, graph_name)
                
                st.subheader("Agent Workflow Graph")
                st.image(graph_image, caption="Agent Workflow")
                # st.image(graph_path, caption="Agent Workflow")
            
            # Run the hedge fund simulation
            result = run_hedge_fund(
                tickers=settings["tickers"],
                start_date=settings["start_date"],
                end_date=settings["end_date"],
                portfolio=portfolio,
                show_reasoning=settings["show_reasoning"],
                selected_analysts=selected_analysts,
                model_name=model_choice,
                model_provider=model_provider,
                progress_bar=progress_bar
            )
            # st.write(result)
            # Display the results
            display_results(result)

if __name__ == "__main__":
    main()