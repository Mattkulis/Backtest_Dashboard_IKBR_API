from ib_insync import *
import pandas as pd
import numpy as np
import datetime as dt
import time
import calendar
from dash import Dash, dcc, html
import plotly.graph_objs as go
import webbrowser
from threading import Timer

# === Connect to IB Gateway or TWS ===
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# === Define start and end dates ===
start_date = dt.date(2024, 1, 1)
end_date = dt.date(2025, 7, 1)
trading_days = pd.bdate_range(start=start_date, end=end_date).to_list()

# === Fetch 1-minute OHLCV for SPY ===
def get_spy_1min_data(date):
    spy = Stock('SPY', 'SMART', 'USD')
    formatted_date = date.strftime('%Y%m%d 16:00:00')
    bars = ib.reqHistoricalData(
        spy,
        endDateTime=formatted_date,
        durationStr='1 D',
        barSizeSetting='1 min',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    df = util.df(bars)
    if df is None or df.empty:
        return None
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date'], inplace=True)
    return df

# === VWAP Calculation ===
def calculate_vwap(df):
    vwap_numerator = (df['close'] * df['volume']).cumsum()
    vwap_denominator = df['volume'].cumsum()
    df['vwap'] = vwap_numerator / vwap_denominator
    return df

# === Entry/Exit Conditions (Short Setup) ===
def detect_entries_exits(df):
    df = df.copy()
    df['ema'] = df['close'].ewm(span=9, adjust=False).mean()
    df['entry'] = False
    df['exit'] = False

    range_minutes = 6
    i = range_minutes
    cutoff_time_reached = False

    while i < len(df) - 10 and not cutoff_time_reached:
        range_df = df.iloc[i - range_minutes:i]
        range_high = range_df['close'].max()
        range_low = range_df['open'].min()
        range_width = range_high - range_low
        range_max_high = range_df['high'].max()
        range_min_low = range_df['low'].min()

        trigger_candle = df.iloc[i]
        if trigger_candle['close'] < range_low - 0.25 * range_width:
            body = abs(trigger_candle['close'] - trigger_candle['open'])
            if body > 1.3 * (range_max_high - range_min_low):
                i += 1
                continue

            next_candle = df.iloc[i + 1]
            entry_time = next_candle.name

            if entry_time.time() > dt.time(10, 0):
                cutoff_time_reached = True
                break

            if next_candle['low'] < trigger_candle['low']:
                stop_loss = range_max_high
                stop_moved = False
                j = i + 2
                closes_above_ema = 0

                df.loc[entry_time, 'entry'] = True

                while j < len(df):
                    bar = df.iloc[j]
                    bar_time = bar.name
                    minutes_since_entry = (bar_time - entry_time).total_seconds() / 60

                    if not stop_moved and minutes_since_entry >= 3:
                        stop_loss = trigger_candle['open']
                        stop_moved = True

                    if bar['high'] >= stop_loss:
                        df.loc[bar_time, 'exit'] = True
                        break

                    if (bar['close'] - bar['ema']) > 0.3 * range_width:
                        df.loc[bar_time, 'exit'] = True
                        break

                    if bar['close'] > bar['ema']:
                        closes_above_ema += 1
                    else:
                        closes_above_ema = 0

                    if closes_above_ema >= 3:
                        df.loc[bar_time, 'exit'] = True
                        break

                    j += 1
                i = j
                continue
        i += 1

    return df

# === Calendar Component ===
def build_calendar_component(daily_pnl_map, year, month):
    first_wd, days_in_month = calendar.monthrange(year, month)
    month_start = dt.date(year, month, 1)
    lead_start = month_start - dt.timedelta(days=first_wd) if first_wd > 0 else month_start
    month_end = dt.date(year, month, days_in_month)

    cells = []
    for i in range(42):
        day = lead_start + dt.timedelta(days=i)
        cells.append(day)
        if day >= month_end and day.weekday() == 6:
            break

    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    header_row = [html.Div(lbl, style={
        'textAlign': 'center',
        'fontWeight': 'bold',
        'padding': '4px',
        'borderBottom': '1px solid #ccc'
    }) for lbl in weekday_labels]

    day_cells = []
    for day in cells:
        in_month = (day.month == month)
        is_weekend = day.weekday() >= 5
        pnl = daily_pnl_map.get(day, None) if in_month else None

        if not in_month or is_weekend or pnl is None:
            bg = '#dcdcdc'
            pnl_str = ''
        else:
            bg = '#b6e8b6' if pnl > 0 else '#f5b5b5' if pnl < 0 else '#dcdcdc'
            pnl_str = f"{pnl:+.2f}"

        day_label = f"{day.day:02d}"
        cell = html.Div([
            html.Div(day_label, style={'fontSize': '10px', 'marginBottom': '2px'}),
            html.Div(pnl_str, style={'fontSize': '12px', 'fontWeight': 'bold'})
        ], style={
            'height': '50px',
            'width': '100px',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center',
            'alignItems': 'center',
            'border': '1px solid #ccc',
            'backgroundColor': bg,
            'margin': '1px'
        })
        day_cells.append(cell)

    grid_children = header_row + day_cells
    calendar_grid = html.Div(
        grid_children,
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(7, 100px)',
            'gridAutoRows': 'minmax(50px, auto)',
            'gap': '0px',
            'width': '100%',
            'maxWidth': '700px',
            'margin': '0 auto'
        }
    )

    month_name = calendar.month_name[month]
    month_header = html.Div(
        f"{month_name} {year}",
        style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '8px', 'fontSize': '16px'}
    )

    return html.Div([month_header, calendar_grid], style={'padding': '10px'})

# === Backtest ===
results = []
executions = []
time_buckets = {'Open': [], 'Noon': [], 'EOD': []}
symbol_pnl = {'SPY': 0.0}

last_reported_month = None

for date in trading_days:
    try:
        if last_reported_month != date.month:
            print(f"Processing data for {date.strftime('%B %Y')}")
            last_reported_month = date.month

        df = get_spy_1min_data(date)
        if df is None or df.empty:
            continue

        df = calculate_vwap(df)
        df = detect_entries_exits(df)

        entry_indices = df.index[df['entry']]
        exit_indices = df.index[df['exit']]

        for entry_time in entry_indices:
            exit_time = exit_indices[exit_indices > entry_time]
            if not exit_time.empty:
                entry_price = df.loc[entry_time]['open']
                exit_time_actual = exit_time[0]
                exit_price = df.loc[exit_time_actual]['close']
                pnl = (entry_price - exit_price) * 100  # short entry, raw dollar PnL
                hold_time = exit_time_actual - entry_time

                hour = entry_time.time().hour
                minute = entry_time.time().minute
                if (hour == 9 and minute >= 30) or (hour == 10) or (hour == 11 and minute <= 30):
                    time_buckets['Open'].append(pnl)
                elif (hour == 11 and minute > 30) or (hour == 12) or (hour == 13):
                    time_buckets['Noon'].append(pnl)
                else:
                    time_buckets['EOD'].append(pnl)

                symbol_pnl['SPY'] += pnl

                results.append({
                    'date': date,
                    'entry_time': entry_time,
                    'exit_time': exit_time_actual,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'hold_time': hold_time
                })

                executions.append(["Entry", entry_time.strftime('%Y-%m-%d %H:%M'), f"{entry_price:.2f}", "100 shares", ""])
                executions.append(["Exit", exit_time_actual.strftime('%Y-%m-%d %H:%M'), f"{exit_price:.2f}", "100 shares", f"{pnl:+.2f}"])

    except Exception as e:
        print(f"Error on {date}: {e}")
        continue

# === Summary ===
df_results = pd.DataFrame(results)

if not df_results.empty:
    df_results.sort_values(by='exit_time', inplace=True)
    df_results['cum_pnl'] = df_results['pnl'].cumsum()

    win_rate = (df_results['pnl'] > 0).mean()
    cum_return = df_results['cum_pnl'].iloc[-1]
    drawdown = (df_results['cum_pnl'].cummax() - df_results['cum_pnl'])
    max_drawdown = drawdown.max()
    calmar = cum_return / max_drawdown if max_drawdown != 0 else 0

    avg_hold_seconds = df_results['hold_time'].mean().total_seconds()
    days, rem = divmod(avg_hold_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    avg_hold_fmt = f"{int(days):02d}:{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    summary_data = {
        "Start Date:": start_date.strftime('%Y-%m-%d'),
        "End Date:": end_date.strftime('%Y-%m-%d'),
        "Total Trades:": len(df_results),
        "Avg Hold Time:": avg_hold_fmt,
        "Win Rate:": f"{win_rate:.2%}",
        "Calmar Ratio:": f"{calmar:.2f}",
        "Cumulative PnL:": f"{cum_return:+.2f}"
    }

    perf_by_time = {
        "Open (9:30–11:30)": f"{np.sum(time_buckets['Open']):+.2f}",
        "Noon (11:31–14:00)": f"{np.sum(time_buckets['Noon']):+.2f}",
        "EOD (14:01–16:00)": f"{np.sum(time_buckets['EOD']):+.2f}"
    }

    perf_by_symbol = [
        ["Ticker", "Symbol", "Cumulative P&L"],
        *[[ticker, ticker, f"{pnl:+.2f}"] for ticker, pnl in symbol_pnl.items()]
    ]

    df_results['exit_date'] = df_results['exit_time'].dt.date
    df_results['exit_weekday'] = df_results['exit_time'].dt.day_name()
    daily_pnl = df_results.groupby('exit_date')['pnl'].sum().to_dict()
    weekday_pnl = df_results.groupby('exit_weekday')['pnl'].sum()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    perf_by_weekday = {day: f"{weekday_pnl.get(day, 0):+.2f}" for day in weekday_order}

    calendar_components = []
    current = dt.date(start_date.year, start_date.month, 1)
    end_month = dt.date(end_date.year, end_date.month, 1)
    while current <= end_month:
        calendar_components.append(build_calendar_component(daily_pnl, current.year, current.month))
        current = dt.date(current.year + 1, 1, 1) if current.month == 12 else dt.date(current.year, current.month + 1, 1)

    app = Dash(__name__)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_results['exit_time'],
        y=df_results['cum_pnl'],
        mode='lines+markers',
        name='Equity Curve',
        line=dict(color='green')
    ))
    fig.update_layout(
        title='Equity Curve (PnL)',
        xaxis_title='Time (EST)',
        yaxis_title='Cumulative PnL',
        hovermode='x unified',
        xaxis_rangeslider_visible=True
    )

    app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'row', 'height': '100vh'}, children=[
        html.Div(children=[
            dcc.Graph(id='equity-curve', figure=fig, style={'height': '70vh'}),
            html.Details([
                html.Summary("📅 Calendar", style={'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Div(calendar_components, style={'maxHeight': '400px', 'overflowY': 'auto'})
            ], style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#f1f1f1'}),
            html.Details([
                html.Summary("📄 Executions", style={'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Table([
                    html.Thead([
                        html.Tr([html.Th(h, style={'fontWeight': 'bold'}) for h in ["Type", "Time", "Price", "Quantity", "P&L"]])
                    ]),
                    html.Tbody([
                        html.Tr([html.Td(cell) for cell in row]) for row in executions
                    ])
                ])
            ], style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#f1f1f1'})
        ], style={'flex': '1', 'overflowY': 'auto', 'padding': '10px'}),
        html.Div(children=[
            html.Details([
                html.Summary("Strategy Results Overview", style={'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Table([
                    html.Tbody([
                        html.Tr([html.Td(k, style={'fontWeight': 'bold'}), html.Td(v)]) for k, v in summary_data.items()
                    ])
                ])
            ]),
            html.Details([
                html.Summary("📆 Performance by Day of Week", style={'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Table([
                    html.Tbody([
                        html.Tr([html.Td(day, style={'fontWeight': 'bold'}), html.Td(pnl)]) for day, pnl in perf_by_weekday.items()
                    ])
                ])
            ]),
            html.Details([
                html.Summary("Performance by Time of Day", style={'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Table([
                    html.Tbody([
                        html.Tr([html.Td(k, style={'fontWeight': 'bold'}), html.Td(v)]) for k, v in perf_by_time.items()
                    ])
                ])
            ]),
            html.Details([
                html.Summary("📈 Performance by Symbol", style={'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Table([
                    html.Thead([
                        html.Tr([html.Th(col, style={'fontWeight': 'bold'}) for col in perf_by_symbol[0]])
                    ]),
                    html.Tbody([
                        html.Tr([html.Td(cell) for cell in row]) for row in perf_by_symbol[1:]
                    ])
                ])
            ])
        ], style={
            'width': '20vw',
            'padding': '10px',
            'borderLeft': '1px solid #ccc',
            'backgroundColor': '#f9f9f9',
            'overflow': 'auto'
        })
    ])

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050")

    Timer(1, open_browser).start()
    app.run()

else:
    print("No trades were taken. Check your entry/exit logic or data availability.")

ib.disconnect()
