import asyncio
import logging
import signal
import sys
from datetime import timedelta
from typing import List, Optional

from tinkoff.invest import AioRequestError, CandleInterval, HistoricCandle, OrderDirection, OrderType
from tinkoff.invest.async_services import AsyncServices
from tinkoff.invest.utils import now

logger = logging.getLogger(__name__)

class BaseStrategy:
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
    
    async def place_order(self, direction: OrderDirection, quantity:int):
        """
        Place order on buy or sell.
        """
        order = await self.client.orders.post_order(
            figi=self.figi,
            quantity=quantity,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET,
            account_id=self.account_id
        )
        logger.info("%s lot. figi=%s", 'Buy' if direction == OrderDirection.ORDER_DIRECTION_BUY else 'Sell', self.figi)

    async def trade(self):
        pass

    async def main_cycle(self):
        """
        Main cycle for live strategy.
        """
        while True:
            try:
                await self.ensure_market_open()
                await self.trade()
            except AioRequestError as are:
                logger.error("Client error %s", are)

            await asyncio.sleep(self.check_interval)

    async def sell_all(self):
        await self.client.cancel_all_orders(account_id=self.account_id)
        logger.info("Cancel all orders. figi=%s", self.figi)
        account = await self.client.users.get_accounts().accounts[0]
        print(account)
        await self.place_order(OrderDirection.ORDER_DIRECTION_BUY, quantity=1)

    async def shutdown(self, signum, frame):
        logging.info("Stop work")
        await self.sell_all()

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