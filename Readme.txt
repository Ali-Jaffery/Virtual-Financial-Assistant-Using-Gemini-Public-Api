# Finance Chatbot Assistant

A finance chatbot assistant built with Streamlit, capable of responding to queries about cryptocurrency prices and various technical analysis indicators. The chatbot can also handle general queries using Google's Generative AI.

## Features

- Get the current price of cryptocurrencies.
- Calculate and return Simple Moving Average (SMA).
- Calculate and return Exponential Moving Average (EMA).
- Calculate and return Relative Strength Index (RSI).
- Calculate and return Moving Average Convergence Divergence (MACD).
- Plot the price history of cryptocurrencies.
- Handle general queries and greetings.
- Use Google's Generative AI for non-finance-related queries.

## Installation

### Prerequisites

- Python 3.8 or later

### Steps

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    ```

4. Set up Google Generative AI:
    Replace `YOUR_GOOGLE_API_KEY` in the code with your actual Google API key:
    ```python
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
    genai.configure(api_key=GOOGLE_API_KEY)
    ```

5. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```
    Replace `app.py` with the name of your Python file.

## Usage

### Example Queries

1. Get the current price of Bitcoin:
    ```txt
    What is the current price of BTC?
    ```

2. Get the 50-day SMA for Ethereum:
    ```txt
    Show me the 50-day SMA for ETH.
    ```

3. Plot the price history of Bitcoin:
    ```txt
    Plot the BTC price chart.
    ```

## Contributing

Please fork the repository and create a pull request to contribute.

## License

This project is licensed under the MIT License.
