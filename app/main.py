import json
from pandas import DataFrame

from dotenv import load_dotenv

import asyncio
import logging
import os

from tinkoff.invest import AsyncClient, CandleInterval, Client, SecurityTradingStatus
from tinkoff.invest.services import InstrumentsService
from tinkoff.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX 
from tinkoff.invest.utils import quotation_to_decimal

from app.strategies.strategy_solver import resolve_strategy


load_dotenv()
TOKEN = os.getenv("TOKEN")
IS_SANDBOX = os.getenv("SANDBOX")

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/base_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_tickers_df():
    """
    Get dataframe with all instruments.
    """
    with Client(TOKEN) as client:
        instruments: InstrumentsService = client.instruments
        tickers = []
        for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
            for item in getattr(instruments, method)().instruments:
                tickers.append(
                    {
                        "name": item.name,
                        "ticker": item.ticker,
                        "class_code": item.class_code,
                        "figi": item.figi,
                        "uid": item.uid,
                        "type": method,
                        "min_price_increment": quotation_to_decimal(
                            item.min_price_increment
                        ),
                        "scale": 9 - len(str(item.min_price_increment.nano)) + 1,
                        "lot": item.lot,
                        "trading_status": str(
                            SecurityTradingStatus(item.trading_status).name
                        ),
                        "api_trade_available_flag": item.api_trade_available_flag,
                        "currency": item.currency,
                        "exchange": item.exchange,
                        "buy_available_flag": item.buy_available_flag,
                        "sell_available_flag": item.sell_available_flag,
                        "short_enabled_flag": item.short_enabled_flag,
                        "klong": quotation_to_decimal(item.klong),
                        "kshort": quotation_to_decimal(item.kshort),
                    }
                )

        tickers_df = DataFrame(tickers)
    return tickers_df

async def run_strategy(portfolio, timeframe, days_back, check_interval):
    """
    From this function starts
    strategy for every ticker from portfolio.
    """
    tickers_df = get_tickers_df()
    target = INVEST_GRPC_API_SANDBOX if IS_SANDBOX else INVEST_GRPC_API 
    async with AsyncClient(token=TOKEN, app_name="TinkoffApp", target=target) as client:
        strategy_tasks = []
        for instrument in portfolio:
            lot = tickers_df[tickers_df["figi"] == instrument["figi"]].iloc[0]["lot"]
            strategy = instrument["strategy"]
            figi = instrument["figi"]
            strategy = resolve_strategy(
                strategy_name=strategy,
                figi=figi,
                timeframe=timeframe,
                check_interval=check_interval,
                days_back=days_back,
                lot=lot,
                client=client
            )
            strategy_tasks.append(asyncio.create_task(strategy.start()))
        await asyncio.gather(*strategy_tasks)


if __name__ == "__main__":
    with open('instruments_config.json') as f:
        data = json.load(f)
    portfolio = list(data["instruments"].values())

    timeframe = CandleInterval.CANDLE_INTERVAL_1_MIN
    days_back = 1
    check_interval = 10 

    loop = asyncio.get_event_loop()
    task = loop.create_task(
        run_strategy(
            portfolio=portfolio,
            timeframe=timeframe,
            days_back=days_back,
            check_interval=check_interval,
        )
    )
    loop.run_until_complete(task)
    