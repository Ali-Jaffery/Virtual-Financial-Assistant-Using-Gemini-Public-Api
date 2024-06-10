import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import nltk
import google.generativeai as genai
import difflib

nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Gemini API Client
def get_crypto_price(symbol):
    response = requests.get(f'https://api.gemini.com/v1/pubticker/{symbol}')
    data = response.json()
    return f"The current price of {symbol} is ${data['last']}."

def calculate_SMA(symbol, window):
    response = requests.get(f'https://api.gemini.com/v2/candles/{symbol}/1day')
    data = pd.DataFrame(response.json(), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    data['close'] = data['close'].astype(float)
    sma = data['close'].rolling(window=window).mean().iloc[-1]
    return f"The {window}-day SMA for {symbol} is ${sma:.2f}."

def calculate_EMA(symbol, window):
    response = requests.get(f'https://api.gemini.com/v2/candles/{symbol}/1day')
    data = pd.DataFrame(response.json(), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    data['close'] = data['close'].astype(float)
    ema = data['close'].ewm(span=window, adjust=False).mean().iloc[-1]
    return f"The {window}-day EMA for {symbol} is ${ema:.2f}."

def calculate_RSI(symbol):
    response = requests.get(f'https://api.gemini.com/v2/candles/{symbol}/1day')
    data = pd.DataFrame(response.json(), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    data['close'] = data['close'].astype(float)
    delta = data['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_up = up.ewm(com=14-1, adjust=False).mean()
    avg_down = down.ewm(com=14, adjust=False).mean()
    rs = avg_up / avg_down
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    return f"The RSI for {symbol} is {rsi:.2f}."

def calculate_MACD(symbol):
    response = requests.get(f'https://api.gemini.com/v2/candles/{symbol}/1day')
    data = pd.DataFrame(response.json(), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    data['close'] = data['close'].astype(float)
    short_EMA = data['close'].ewm(span=12, adjust=False).mean()
    long_EMA = data['close'].ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f"The MACD for {symbol} is {MACD.iloc[-1]:.2f}, the signal line is {signal.iloc[-1]:.2f}, and the MACD histogram is {MACD_histogram.iloc[-1]:.2f}."

def plot_crypto_price(symbol):
    response = requests.get(f'https://api.gemini.com/v2/candles/{symbol}/1day')
    data = pd.DataFrame(response.json(), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    data['close'] = data['close'].astype(float)
    data['time'] = pd.to_datetime(data['time'], unit='ms')
    plt.figure(figsize=(10, 5))
    plt.plot(data['time'], data['close'])
    plt.title(f'{symbol} Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.savefig('crypto.png')
    plt.close()
    return 'crypto.png'

available_functions = {
    'price': get_crypto_price,
    'sma': calculate_SMA,
    'ema': calculate_EMA,
    'rsi': calculate_RSI,
    'macd': calculate_MACD,
    'plot': plot_crypto_price
}

def parse_input(user_input):
    tokens = word_tokenize(user_input.lower())
    symbol = None
    window = None
    func_name = None
    for token in tokens:
        if token in available_functions:
            func_name = token
        elif token in ['btc', 'eth']:
            symbol = 'btcusd' if token == 'btc' else 'ethusd'
        elif token.isdigit():
            window = int(token)
    
    if func_name in ['price', 'rsi', 'macd', 'plot']:
        if symbol:
            return func_name, symbol, None
    elif func_name in ['sma', 'ema']:
        if symbol and window:
            return func_name, symbol, window
    
    return None, None, None

# Check if the question is finance-related
def is_finance_related(query):
    finance_keywords = [
        "price", "sma", "ema", "rsi", "macd", "plot", "btc", "eth", "crypto", 
        "finance", "stock", "market", "investment", "trade", "trading", 
        "portfolio", "financial", "earnings", "revenue", "profit", "loss", 
        "valuation", "equity", "debt", "capital", "business", "economy", "analysis",
        "budget", "balance sheet", "cash flow", "dividend", "forecast", "index", 
        "inflation", "interest rate", "liquidity", "merger", "acquisition", 
        "return", "risk", "share", "stock price", "bond", "commodity", 
        "currency", "derivative", "exchange", "fund", "growth", "income", 
        "leverage", "option", "principal", "yield", "asset", "liability", 
        "recession", "recovery", "supply", "demand", "tax", "regulation", 
        "audit", "compliance", "strategy", "market cap", "ipo", "shareholder", 
        "stakeholder", "underwriting", "venture capital", "angel investor", 
        "private equity", "hedge fund", "mutual fund", "exchange-traded fund"
    ]
    tokens = word_tokenize(query.lower())
    return any(difflib.get_close_matches(token, finance_keywords, cutoff=0.5) for token in tokens) #check similarity level of words

# Configure generative AI
GOOGLE_API_KEY = "AIzaSyDLsOJWNKeCnhyfwaoqI5dErB5io8BuL4M"
genai.configure(api_key=GOOGLE_API_KEY)
geminiModel = genai.GenerativeModel("gemini-pro")
chat = geminiModel.start_chat(history=[])

def get_gemini_response(query):
    instantResponse = chat.send_message(query, stream=True)
    return instantResponse

# Function to handle greetings and creator information
def handle_general_queries(query):
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    creator_queries = ["who created you", "who made you", "who is your creator"]

    tokens = word_tokenize(query.lower())
    if any(greeting in tokens for greeting in greetings):
        return "Hello! How can I assist you with your business and finance questions today?"
    elif any(creator_query in query.lower() for creator_query in creator_queries):
        return "I am a virtual finance assistant created by Kashan, Eima, and Yahya for their AI project."
    else:
        return None

# Streamlit App
st.title("Finance Chatbot Assistant")

if 'chat_sessions' not in st.session_state:
    st.session_state['chat_sessions'] = {"Session 1": []}
    st.session_state['current_session'] = "Session 1"

# Sidebar for managing chat sessions
st.sidebar.title("Chat Sessions")
session_names = list(st.session_state['chat_sessions'].keys())

selected_session = st.sidebar.selectbox("Select a session", session_names)
if st.sidebar.button("Start a new session"):
    new_session_name = f"Session {len(st.session_state['chat_sessions']) + 1}"
    st.session_state['chat_sessions'][new_session_name] = []
    st.session_state['current_session'] = new_session_name
    st.experimental_rerun()

st.session_state['current_session'] = selected_session

st.sidebar.subheader("Current Session")
st.sidebar.write(st.session_state['current_session'])

user_input = st.text_input('Your input:', key='input_text')
submit_button = st.button('Enter', key='enter_button')

# Commonly asked questions
common_questions = [
    "What is the current price of BTC?",
    "Show me the RSI for ETH",
    "Plot the BTC price chart",
    "What is the 50-day EMA for BTC?"
]

st.write("Common Questions:")
for question in common_questions:
    if st.button(question):
        user_input = question
        submit_button = True

if submit_button and user_input:
    try:
        st.session_state['chat_sessions'][st.session_state['current_session']].append({'role': 'user', 'content': user_input})
        
        general_response = handle_general_queries(user_input)
        if general_response:
            function_response = general_response
        elif not is_finance_related(user_input):
            function_response = "This is an irrelevant question. Please ask a finance-related question."
        else:
            func_name, symbol, window = parse_input(user_input)
            
            if func_name:
                function_to_call = available_functions[func_name]
                if func_name in ['price', 'rsi', 'macd', 'plot'] and symbol:
                    function_response = function_to_call(symbol)
                elif func_name in ['sma', 'ema'] and symbol and window:
                    function_response = function_to_call(symbol, window)
                else:
                    function_response = "Unable to parse the input correctly. Please check your input."
            else:
                function_response = get_gemini_response(user_input)
                function_response = ''.join([chunk.text for chunk in function_response])

        st.session_state['chat_sessions'][st.session_state['current_session']].append({'role': 'assistant', 'content': function_response})

        if func_name == 'plot' and symbol:
            st.image('crypto.png')
        else:
            st.text(function_response)

    except Exception as e:
        st.text(f"Error occurred: {str(e)}")

st.subheader("Chat History")
for message in st.session_state['chat_sessions'][st.session_state['current_session']]:
    st.write(f"{message['role']}: {message['content']}")