import yfinance as yf
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np

# --- Data Fetching Functions ---

# Function to fetch historical data for a given ticker
def get_historical_data(ticker, start=None, end=None, interval='1d', max_retries=3):
    for attempt in range(max_retries):
        try:
            return yf.download(ticker, start=start, end=end, interval=interval, actions=False, progress=False)
        except Exception as e:
            print(f"Attempt {attempt + 1}: Failed to download {ticker} data due to {e}. Retrying...")
            time.sleep(2)
    raise Exception(f"Failed to download {ticker} data after {max_retries} attempts.")

# Function to fetch portfolio data from CSV and calculate its total value over time
def get_portfolio_data(filepath, start_date, end_date):
    portfolio_df = pd.read_csv(filepath)
    tickers = portfolio_df['Ticker'].unique()
    shares = portfolio_df.set_index('Ticker')['Shares'].to_dict()

    portfolio_value_df = pd.DataFrame()

    for ticker in tickers:
        ticker_data = get_historical_data(ticker, start=start_date, end=end_date)
        ticker_data['Value'] = ticker_data['Close'] * shares[ticker]
        if portfolio_value_df.empty:
            portfolio_value_df = ticker_data[['Value']].copy()
        else:
            portfolio_value_df['Value'] += ticker_data['Value']

    return portfolio_value_df

# --- App Layout and Components ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

# Dropdown options for the top 50 stocks in the S&P 500
dropdown_options = [
    {"label": "Apple Inc.", "value": "AAPL"},
    {"label": "Microsoft Corporation", "value": "MSFT"},
    {"label": "Amazon.com, Inc.", "value": "AMZN"},
    {"label": "NVIDIA Corporation", "value": "NVDA"},
    {"label": "Alphabet Inc.", "value": "GOOGL"},
    {"label": "Tesla, Inc.", "value": "TSLA"},
    {"label": "Berkshire Hathaway Inc.", "value": "BRK.B"},
    {"label": "Johnson & Johnson", "value": "JNJ"},
    {"label": "Meta Platforms, Inc.", "value": "META"},
    {"label": "UnitedHealth Group Incorporated", "value": "UNH"},

    # Top Index Funds
    {"label": "S&P 500", "value": "^GSPC"},
    {"label": "Dow Jones Industrial Average", "value": "^DJI"},
    {"label": "Nasdaq Composite", "value": "^IXIC"},
    {"label": "Russell 2000", "value": "^RUT"},
    {"label": "FTSE 100", "value": "^FTSE"},
    {"label": "DAX", "value": "^GDAXI"},
    {"label": "CAC 40", "value": "^FCHI"},
    {"label": "Nikkei 225", "value": "^N225"},
    {"label": "Hang Seng", "value": "^HSI"},
    {"label": "Shanghai Composite", "value": "^SSEC"},

    # Top Stocks
    {"label": "Visa Inc.", "value": "V"},
    {"label": "Procter & Gamble Co.", "value": "PG"},
    {"label": "The Home Depot, Inc.", "value": "HD"},
    {"label": "The Walt Disney Company", "value": "DIS"},
    {"label": "Mastercard Incorporated", "value": "MA"},
    {"label": "Exxon Mobil Corporation", "value": "XOM"},
    {"label": "PepsiCo, Inc.", "value": "PEP"},
    {"label": "Coca-Cola Company", "value": "KO"},
    {"label": "Verizon Communications Inc.", "value": "VZ"},
    {"label": "Cisco Systems, Inc.", "value": "CSCO"},
    {"label": "Netflix, Inc.", "value": "NFLX"},
    {"label": "AT&T Inc.", "value": "T"},
    {"label": "Intel Corporation", "value": "INTC"},
    {"label": "NIKE, Inc.", "value": "NKE"},
    {"label": "Comcast Corporation", "value": "CMCSA"},
    {"label": "Merck & Co., Inc.", "value": "MRK"},
    {"label": "Thermo Fisher Scientific Inc.", "value": "TMO"},
    {"label": "International Business Machines Corporation", "value": "IBM"},
    {"label": "Abbott Laboratories", "value": "ABT"},
    {"label": "Broadcom Inc.", "value": "AVGO"},
    {"label": "Amgen Inc.", "value": "AMGN"},
    {"label": "QUALCOMM Incorporated", "value": "QCOM"},
    {"label": "Salesforce, Inc.", "value": "CRM"},
    {"label": "Medtronic plc", "value": "MDT"},
    {"label": "Texas Instruments Incorporated", "value": "TXN"},
    {"label": "Lockheed Martin Corporation", "value": "LMT"},
    {"label": "Starbucks Corporation", "value": "SBUX"},
    {"label": "Honeywell International Inc.", "value": "HON"},
    {"label": "Costco Wholesale Corporation", "value": "COST"},
    {"label": "S&P Global Inc.", "value": "SPGI"},
    {"label": "ServiceNow, Inc.", "value": "NOW"},
    {"label": "Bristol Myers Squibb Company", "value": "BMY"},
    {"label": "Intuitive Surgical, Inc.", "value": "ISRG"},
    {"label": "Gilead Sciences, Inc.", "value": "GILD"},
    {"label": "Stryker Corporation", "value": "SYK"},
    {"label": "Air Products and Chemicals, Inc.", "value": "APD"},
    {"label": "Adobe Inc.", "value": "ADBE"},
    {"label": "Philip Morris International Inc.", "value": "PM"},
    {"label": "Mondelez International, Inc.", "value": "MDLZ"},
    {"label": "Danaher Corporation", "value": "DHR"},
    {"label": "Caterpillar Inc.", "value": "CAT"},
    {"label": "Newmont Corporation", "value": "NEM"},
    {"label": "Cigna Corporation", "value": "CI"},
    {"label": "Novartis AG", "value": "NVS"},
    {"label": "W.W. Grainger, Inc.", "value": "GWW"},
    {"label": "Pfizer Inc.", "value": "PFE"},
    {"label": "Johnson Controls International plc", "value": "JCI"},
    {"label": "Chevron Corporation", "value": "CVX"},
    {"label": "3M Company", "value": "MMM"},
    {"label": "American Express Company", "value": "AXP"},
    {"label": "Colgate-Palmolive Company", "value": "CL"},
    {"label": "Boeing Company", "value": "BA"},
    {"label": "Anthem, Inc.", "value": "ANTM"},
    {"label": "Biogen Inc.", "value": "BIIB"},
    {"label": "Ford Motor Company", "value": "F"},
    {"label": "General Motors Company", "value": "GM"},
    {"label": "BlackRock, Inc.", "value": "BLK"},
    {"label": "Eli Lilly and Company", "value": "LLY"},
    {"label": "McDonald's Corporation", "value": "MCD"},
    {"label": "PayPal Holdings, Inc.", "value": "PYPL"},
    {"label": "Booking Holdings Inc.", "value": "BKNG"},
    {"label": "Marriott International, Inc.", "value": "MAR"},
    {"label": "Walmart Inc.", "value": "WMT"},
    {"label": "Target Corporation", "value": "TGT"},
    {"label": "General Electric Company", "value": "GE"}
]


# --- Callback Functions for Graph Updates ---

@app.callback(
    [Output('percent-change-graph', 'figure'),
     Output('weekly-percent-change-graph', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('index-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graphs(selected_index, start_date, end_date):
    # Check if selected_index and date range are valid
    if not selected_index or not start_date or not end_date:
        empty_fig = {'data': [], 'layout': go.Layout(title='Invalid Input')}
        return empty_fig, empty_fig

    # Fetch stock and portfolio data
    stock_data = get_historical_data(ticker=selected_index, start=start_date, end=end_date)
    portfolio_data = get_portfolio_data('/Users/maiaseidel/Desktop/portfolio.csv', start_date, end_date)

    # Check if data is empty
    if stock_data.empty or portfolio_data.empty:
        empty_fig = {'data': [], 'layout': go.Layout(title='No Data Available')}
        return empty_fig, empty_fig

    # Cumulative Percent Change Calculation
    stock_data['Percent Change'] = (stock_data['Close'] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0] * 100
    portfolio_data['Percent Change'] = (portfolio_data['Value'] - portfolio_data['Value'].iloc[0]) / portfolio_data['Value'].iloc[0] * 100

    # Graph 1: Cumulative Percent Change
    figure1 = {
        'data': [
            go.Scatter(x=stock_data.index, y=stock_data['Percent Change'], mode='lines', name=f'{selected_index} Percent Change'),
            go.Scatter(x=portfolio_data.index, y=portfolio_data['Percent Change'], mode='lines', name='Portfolio Percent Change')
        ],
        'layout': go.Layout(title=f'{selected_index} vs. Portfolio Percent Change', xaxis={'title': 'Date'}, yaxis={'title': 'Percent Change (%)'})
    }

    # Weekly Percent Change Calculation
    stock_weekly_pct_change = stock_data['Close'].resample('W').last().pct_change() * 100
    portfolio_weekly_pct_change = portfolio_data['Value'].resample('W').last().pct_change() * 100

    combined_data = pd.concat([stock_weekly_pct_change, portfolio_weekly_pct_change], axis=1, keys=['Index', 'Portfolio']).dropna()
    outperform_mask = combined_data['Portfolio'] > combined_data['Index']

    # Highlighting outperformance/underperformance regions
    shapes = create_performance_shapes(combined_data, outperform_mask)

    # Graph 2: Weekly Percent Change
    figure2 = {
        'data': [
            go.Scatter(x=combined_data.index, y=combined_data['Index'], mode='lines', name=f'{selected_index} Weekly Percent Change'),
            go.Scatter(x=combined_data.index, y=combined_data['Portfolio'], mode='lines', name='Portfolio Weekly Percent Change')
        ],
        'layout': go.Layout(title=f'{selected_index} vs. Portfolio Weekly Percent Change', xaxis={'title': 'Date'}, yaxis={'title': 'Weekly Percent Change (%)'}, shapes=shapes)
    }
    # Fetch portfolio data for pie chart
    portfolio_df = pd.read_csv('/Users/maiaseidel/Desktop/portfolio.csv')
    tickers = portfolio_df['Ticker'].unique()
    shares = portfolio_df.set_index('Ticker')['Shares'].to_dict()
    pie_values = []
    pie_labels = []

    for ticker in tickers:
        stock_data = get_historical_data(ticker, start=start_date, end=end_date)
        current_value = stock_data['Close'].iloc[-1] * shares[ticker]
        pie_values.append(current_value)
        pie_labels.append(ticker)

    # Create pie chart
    pie_chart = {
        'data': [go.Pie(labels=pie_labels, values=pie_values, hole=0.3)],
        'layout': go.Layout(title='Portfolio Stock Weights')
    }

    return figure1, figure2, pie_chart

# Helper function to create performance shapes for the graph
def create_performance_shapes(combined_data, outperform_mask):
    shapes = []
    section_start = combined_data.index[0]
    previous_mask = outperform_mask.iloc[0]

    for i in range(1, len(outperform_mask)):
        current_mask = outperform_mask.iloc[i]
        if current_mask != previous_mask:
            color_intensity = min(abs(combined_data['Portfolio'].iloc[i] - combined_data['Index'].iloc[i]) / 5, 1)
            shape_color = f'rgba(0, 255, 0, {color_intensity})' if previous_mask else f'rgba(255, 0, 0, {color_intensity})'
            shapes.append({
                'type': 'rect', 'xref': 'x', 'yref': 'paper',
                'x0': section_start, 'x1': combined_data.index[i],
                'y0': 0, 'y1': 1, 'fillcolor': shape_color, 'opacity': 0.3, 'line': {'width': 0}
            })
            section_start = combined_data.index[i]
            previous_mask = current_mask

    return shapes


def calculate_metrics(portfolio_returns, market_returns, risk_free_rate=0.01):
    # Calculate metrics
    annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
    cumulative_return = (1 + portfolio_returns).prod() - 1
    volatility = portfolio_returns.std() * np.sqrt(252)

    winning_days = portfolio_returns[portfolio_returns > 0].count()
    total_days = portfolio_returns.count()
    winning_day_ratio = winning_days / total_days if total_days > 0 else 0

    # Calculate Beta
    covariance = np.cov(portfolio_returns, market_returns)[0][1]
    market_variance = market_returns.var()
    beta = covariance / market_variance if market_variance != 0 else 0

    # Calculate Alpha
    portfolio_return = portfolio_returns.mean() * 252  # annualize
    market_return = market_returns.mean() * 252  # annualize
    alpha = portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))

    return {
        "Annual Return": annual_return,
        "Cumulative Return": cumulative_return,
        "Volatility": volatility,
        "Winning Day Ratio": winning_day_ratio,
        "Alpha": alpha,
        "Beta": beta
    }

@app.callback(
    Output('metrics-output', 'children'),
    [Input('index-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_metrics(selected_index, start_date, end_date):
    if not selected_index or not start_date or not end_date:
        return "Select an index and date range to see metrics."

    # Fetch portfolio data and calculate returns
    portfolio_data = get_portfolio_data('/Users/maiaseidel/Desktop/portfolio.csv', start_date, end_date)
    portfolio_returns = portfolio_data['Value'].pct_change().dropna()  # Calculate daily returns
    market_data = get_historical_data(selected_index, start_date, end_date)
    market_returns = market_data['Close'].pct_change().dropna()  # Calculate daily returns

    metrics = calculate_metrics(portfolio_returns, market_returns)

    # Return formatted metrics as a string
    return html.Div([
        html.Div([
            html.Span("Annual Return: ", style={'fontWeight': 'bold'}),
            html.Span(f"{metrics['Annual Return']:.2%}")
        ]),
        html.Div([
            html.Span("Cumulative Return: ", style={'fontWeight': 'bold'}),
            html.Span(f"{metrics['Cumulative Return']:.2%}")
        ]),
        html.Div([
            html.Span("Volatility: ", style={'fontWeight': 'bold'}),
            html.Span(f"{metrics['Volatility']:.2%}")
        ]),
        html.Div([
            html.Span("Winning Day Ratio: ", style={'fontWeight': 'bold'}),
            html.Span(f"{metrics['Winning Day Ratio']:.2%}")
        ]),
        html.Div([
            html.Span("Alpha: ", style={'fontWeight': 'bold'}),
            html.Span(f"{metrics['Alpha']:.2f}")
        ]),
        html.Div([
            html.Span("Beta: ", style={'fontWeight': 'bold'}),
            html.Span(f"{metrics['Beta']:.2f}")
        ])
    ])

@app.callback(
    Output('suggestion-output', 'children'),
    [Input('index-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def suggest_portfolio_changes(selected_index, start_date, end_date):
    if not selected_index or not start_date or not end_date:
        return "Select an index and date range to get portfolio suggestions."

    # Fetch portfolio and market data
    portfolio_data = get_portfolio_data('/Users/maiaseidel/Desktop/portfolio.csv', start_date, end_date)
    stock_data = get_historical_data(ticker=selected_index, start=start_date, end=end_date)

    if portfolio_data.empty or stock_data.empty:
        return "No data available for suggestions."

    # Logic to suggest changes (basic example)
    stock_pct_change = stock_data['Close'].pct_change().mean() * 100
    portfolio_pct_change = portfolio_data['Value'].pct_change().mean() * 100

    suggestions = []
    if portfolio_pct_change < stock_pct_change:
        suggestions.append("Consider increasing your stock holdings to improve portfolio performance.")
    else:
        suggestions.append("Your portfolio is outperforming the index! Review stocks for any over-reliance on top performers.")

    return html.Ul([html.Li(suggestion) for suggestion in suggestions])

# Define the layout for the app
app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H1("Performance Dashboard"), className="mb-4")),

        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id='index-dropdown',
                        options=dropdown_options,
                        value='^GSPC',  # Default selection
                        clearable=False,
                        style={'color': 'black', 'fontWeight': 'bold'},
                        className="mb-4"
                    ), width=4
                ),


                dbc.Col(
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=(datetime.now() - timedelta(days=365)).date(),
                        end_date=datetime.now().date(),
                        display_format='YYYY-MM-DD',
                        className="mb-4"
                    ), width=4
                ),
                dbc.Col(
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button('Upload Portfolio CSV', style={'color': 'black', 'fontWeight': 'bold'}),
                        multiple=False,  # Allow only one file at a time
                    ), width=4  # Adjust the width as needed
                ),

            ]
        ),

        dbc.Row([dbc.Col(dcc.Graph(id='percent-change-graph'), width=12)]),
        dbc.Row([dbc.Col(dcc.Graph(id='weekly-percent-change-graph'), width=12)]),
        dbc.Row([dbc.Col(dcc.Graph(id='pie-chart'), width=12)]),

        dbc.Row(
            dbc.Col(

                html.Div("Your Portfolio Stats:", style={'textAlign': 'center', 'fontSize': '24px'}),
                # Center and size the text
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(style={'height': '24px'}),  # Adjust the height as needed for spacing
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(

                html.Div(id='metrics-output', style={'textAlign': 'center', 'fontSize': '24px'}),  # Center and size the text
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(style={'height': '24px'}),  # Adjust the height as needed for spacing
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(

                html.Div("Smart Analysis of Your Portfolio:", style={'textAlign': 'center', 'fontSize': '24px'}),
                # Center and size the text
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(style={'height': '24px'}),  # Adjust the height as needed for spacing
                width=12
            )
        ),
        dbc.Col(html.Div(id='suggestion-output', style={'textAlign': 'center', 'fontSize': '18px'}), width=12)
    ],
    fluid=True
)
# --- Main ---

if __name__ == '__main__':
    app.run_server(debug=True)