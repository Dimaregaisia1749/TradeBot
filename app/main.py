import json
from pandas import DataFrame

from dotenv import load_dotenv

import asyncio
import logging
import os
from datetime import timedelta
from typing import List, Optional

from tinkoff.invest import AioRequestError, AsyncClient, CandleInterval, HistoricCandle, OrderDirection, OrderType, Quotation, Client, SecurityTradingStatus
from tinkoff.invest.services import InstrumentsService
from tinkoff.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX 
from tinkoff.invest.async_services import AsyncServices
from tinkoff.invest.utils import now, quotation_to_decimal


load_dotenv()
TOKEN = os.getenv("TOKEN")
IS_SANDBOX = os.getenv("SANDBOX")

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ABCStrategy:
    """
    This class is responsible for a strategy.
    """
    def __init__(
        self,
        figi: str,
        timeframe: CandleInterval,
        days_back: int,
        check_interval: int,
        lot: int,
        client: Optional[AsyncServices],
    ):
        self.account_id = None
        self.figi = figi
        self.timeframe = timeframe
        self.days_back = days_back
        self.lot = lot
        self.check_interval = check_interval
        self.client = client
        self.candles: List[HistoricCandle] = []

    async def get_historical_data(self):
        """
        Gets historical data for the instrument. Returns list of candles.
        Requests all the candles of timeframe from days_back to now.

        :return: list of HistoricCandle
        """
        logger.debug(
            "Start getting historical data for %s days back from now. figi=%s",
            self.days_back,
            self.figi,
        )
        async for candle in self.client.get_all_candles(
            figi=self.figi,
            from_=now() - timedelta(days=self.days_back),
            to=now(),
            interval=self.timeframe,
        ):
            if candle not in self.candles:
                if candle.is_complete:
                    self.candles.append(candle)
                    logger.debug("Found %s - figi=%s", candle, self.figi)

    async def ensure_market_open(self):
        """
        Ensure that the market is open. Loop until the instrument is available.
        :return: when instrument is available for trading.
        """
        trading_status = await self.client.market_data.get_trading_status(
            figi=self.figi
        )
        while not (
            trading_status.market_order_available_flag
            and trading_status.api_trade_available_flag
        ):
            logger.debug("Waiting for the market to open. figi=%s", self.figi)
            await asyncio.sleep(60)
            trading_status = await self.client.market_data.get_trading_status(
                figi=self.figi
            )

    async def get_price(self):
        """
        Get current price.
        """
        market_data = await self.client.market_data.get_last_prices(figi=[self.figi])
        price = market_data.last_prices[0].price
        return price.units + price.nano / 1e9
    
    async def place_order(self, direction: OrderDirection):
        """
        Place order on buy or sell.
        """
        order = await self.client.orders.post_order(
            figi=self.figi,
            quantity=self.lot,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET,
            account_id=self.account_id
        )
        logging.info(f"Deal {'buy' if direction == OrderDirection.ORDER_DIRECTION_BUY else 'sell'} completed.")

    async def trade(self):
        """
        Decision maker.
        """
        logging.info("Start ABC. figi=%s", self.figi)

        prev_price = await self.get_price()
        while True:
            await asyncio.sleep(60)
            curr_price = await self.get_price()
            logger.info("Price. figi=%s price=%s", self.figi, curr_price)
            if curr_price > prev_price * 1.001:
                await self.place_order(OrderDirection.ORDER_DIRECTION_BUY)
                logger.info("Buy lot. figi=%s", self.figi)
            elif curr_price < prev_price * 0.999:
                await self.place_order(OrderDirection.ORDER_DIRECTION_SELL)
                logger.info("Sell lot. figi=%s", self.figi)

            prev_price = curr_price

    async def main_cycle(self):
        """
        Main cycle for live strategy.
        """
        while True:
            try:
                await self.ensure_market_open()
                await self.place_order(OrderDirection.ORDER_DIRECTION_BUY)
                logger.info("Buy lot. figi=%s", self.figi)
                await self.trade()
            except AioRequestError as are:
                logger.error("Client error %s", are)

            await asyncio.sleep(self.check_interval)

    async def start(self):
        """
        Strategy starts from this function.
        """
        if self.account_id is None:
            try:
                self.account_id = (
                    (await self.client.users.get_accounts()).accounts.pop().id
                )
            except AioRequestError as are:
                logger.error("Error taking account id. Stopping strategy. %s", are)
                return
        await self.main_cycle()

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
            lot = tickers_df[tickers_df["figi"] == instrument].iloc[0]["lot"]
            strategy = ABCStrategy(
                figi=instrument,
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
    