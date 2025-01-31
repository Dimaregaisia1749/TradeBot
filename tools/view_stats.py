import pandas as pd
import os
import asyncio

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tinkoff.invest import AsyncClient, OperationState, OperationType
from tinkoff.invest import GetOperationsByCursorRequest
from tinkoff.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from tinkoff.invest.utils import quotation_to_decimal
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(override=True)
TOKEN = os.getenv("TOKEN")
IS_SANDBOX = True if os.getenv("SANDBOX") == 'True' else False

async def fetch_operations():
    target = INVEST_GRPC_API_SANDBOX if IS_SANDBOX else INVEST_GRPC_API 
    async with AsyncClient(token=TOKEN, target=target) as client:
        sandbox_account = (await client.users.get_accounts()).accounts[0]
        account_id = sandbox_account.id

        operations = []
        cursor = None
        
        while True:
            response = await client.operations.get_operations_by_cursor(GetOperationsByCursorRequest(
                account_id=account_id,
                cursor=cursor,
                state=OperationState.OPERATION_STATE_EXECUTED,
                limit=1000
            ))
            operations.extend(response.items)
            if not response.has_next:
                break
            cursor = response.next_cursor
        return operations

def process_operations(operations):
    """
    Convert operations to df
    """
    data = []
    for op in operations:
        if op.type not in [OperationType.OPERATION_TYPE_BUY, OperationType.OPERATION_TYPE_SELL]:
            continue
        data.append({
            "timestamp": op.date.astimezone(ZoneInfo("Europe/Moscow")),
            "type": "BUY" if op.type == OperationType.OPERATION_TYPE_BUY else "SELL",
            "figi": op.figi,
            "quantity": op.quantity,
            "price": quotation_to_decimal(op.price),
            "volume": quotation_to_decimal(op.price) * op.quantity
        })
    
    return pd.DataFrame(data)

def create_portfolio_visualization(portfolio_df, trades_df):
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.05,
        column_widths=[0.8, 0.2],
        specs=[
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "domain"}]
        ],
        subplot_titles=("Стоимость портфеля", "Баланс", "Доходность", "Сводка")
    )

    fig.update_layout(grid={"rows": 3, "columns": 2, "pattern": "independent"})

    #Portfolio
    fig.add_trace(
        go.Scatter(
            x=portfolio_df['timestamp'],
            y=portfolio_df['portfolio_value'],
            name="Общая стоимость",
            mode='lines',
            line=dict(color='#2c3e50')
        ),
        row=1, col=1
    )

    buys = trades_df[trades_df['type'] == 'BUY']
    sells = trades_df[trades_df['type'] == 'SELL']
    
    fig.add_trace(
        go.Scatter(
            x=buys['timestamp'],
            y=[portfolio_df['portfolio_value'].max()]*len(buys),
            mode='markers',
            name='Покупки',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            hovertext=buys.apply(lambda r: f"{r['figi']}<br>{r['quantity']} шт.", axis=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sells['timestamp'],
            y=[portfolio_df['portfolio_value'].max()]*len(sells),
            mode='markers',
            name='Продажи',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            hovertext=sells.apply(lambda r: f"{r['figi']}<br>{r['quantity']} шт.", axis=1)
        ),
        row=1, col=1
    )

    #Balance
    fig.add_trace(
        go.Bar(
            x=portfolio_df['timestamp'],
            y=portfolio_df['balance'],
            name="Баланс",
            marker_color='#18bc9c'
        ),
        row=2, col=1
    )

    #Profit
    yield_df = portfolio_df.copy()
    yield_df['positive'] = yield_df['expected_yield'] >= 0

    positive = yield_df[yield_df['positive']]
    negative = yield_df[~yield_df['positive']]

    #Fill for positive
    fig.add_trace(go.Scatter(
        x=positive['timestamp'],
        y=positive['expected_yield'],
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(40, 167, 69, 0.2)',
        name='Positive Yield',
        showlegend=False
    ), row=3, col=1)

    #Fill for negative
    fig.add_trace(go.Scatter(
        x=negative['timestamp'],
        y=negative['expected_yield'],
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(220, 53, 69, 0.2)',
        name='Negative Yield',
        showlegend=False
    ), row=3, col=1)

    #Main yield
    fig.add_trace(go.Scatter(
        x=yield_df['timestamp'],
        y=yield_df['expected_yield'],
        line=dict(color='#444'),
        name='Доходность',
        hoverinfo='x+y'
    ), row=3, col=1)

    #Side summary
    last = portfolio_df.iloc[-1]
    summary_stats = {
        "Общая стоимость": f"{last['portfolio_value']:,.2f} ₽",
        "Текущий баланс": f"{last['balance']:,.2f} ₽",
        "Суммарная доходность": f"{last['expected_yield']:+,.4f} %",
        "Всего сделок": f"{len(trades_df):,}",
        "Покупки/Продажи": f"{len(trades_df[trades_df['type'] == 'BUY']):,}/{len(trades_df[trades_df['type'] == 'SELL']):,}"
    }

    #Summary text
    fig.add_annotation(
        x=1.2,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>Итоговая сводка:</b><br>" + "<br>".join(
            [f"{k}: {v}" for k, v in summary_stats.items()]
        ),
        showarrow=False,
        align="left",
        font=dict(size=12),
        bordercolor="black",
        borderwidth=1,
        borderpad=10,
        bgcolor="white"
    )

    #Layout settings
    fig.update_layout(
        title_text="Анализ портфеля",
        template="plotly_white",
        hovermode="x unified",
        height=1000,
        margin=dict(r=300),  # Место для сводки
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Стоимость, ₽", row=1, col=1)
    fig.update_yaxes(title_text="Баланс, ₽", row=2, col=1)
    fig.update_yaxes(title_text="Доходность, ₽", row=3, col=1)
    fig.update_xaxes(title_text="Дата", row=3, col=1)
    
    return fig

async def main():
    portfolio_df = pd.read_csv('sandbox_logs/portfolio_log.csv', parse_dates=['timestamp'])
    operations = await fetch_operations()
    trades_df = process_operations(operations)

    fig = create_portfolio_visualization(portfolio_df, trades_df)
    
    fig.write_html("portfolio_dashboard.html")
    print("Дашборд сохранен в portfolio_dashboard.html")

if __name__ == "__main__":
    asyncio.run(main())