import asyncio
import logging
from datetime import timedelta
from typing import List, Optional

from tinkoff.invest import AioRequestError, CandleInterval, HistoricCandle, OrderDirection, OrderType
from tinkoff.invest.async_services import AsyncServices
from tinkoff.invest.utils import now
from tinkoff.invest.utils import quotation_to_decimal

logger = logging.getLogger(__name__)

class BaseStrategy:
    """
    Class for base instruments for strategy
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
        price = float(quotation_to_decimal(market_data.last_prices[0].price))
        return price
    
    async def place_order(self, direction: OrderDirection, quantity:int):
        """
        Place order on buy or sell.
        """
        current_amount = await self.get_position_quantity()
        if direction == OrderDirection.ORDER_DIRECTION_SELL and current_amount < quantity:
            logger.info("Can't sell lot. figi=%s. Current amount:%s lower than sell amount:%s", self.figi, current_amount, quantity)
            return
        order = await self.client.orders.post_order(
            figi=self.figi,
            quantity=quantity,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET,
            account_id=self.account_id
        )
        logger.info("%s lot. figi=%s", 'Buy' if direction == OrderDirection.ORDER_DIRECTION_BUY else 'Sell', self.figi)
        return float(quotation_to_decimal(order.total_order_amount))

    async def trade(self):
        pass

    async def main_cycle(self):
        """
        Main cycle for live strategy.
        """
        while not(now().second == 0 and now().minute % 5 == 0):
            pass
        while True:
            try:
                await self.ensure_market_open()
                await self.trade()
            except AioRequestError as are:
                logger.error("Client error %s", are)
            await asyncio.sleep(self.check_interval)
    
    async def get_position_quantity(self) -> int:
        """
        Get quantity of the instrument in the position.
        :return: int - quantity
        """
        positions = (await self.client.operations.get_portfolio(account_id=self.account_id)).positions
        for position in positions:
            if position.figi == self.figi:
                return int(quotation_to_decimal(position.quantity))
        return 0

    async def sell_all(self):
        trading_status = await self.client.market_data.get_trading_status(
            figi=self.figi
        )
        if(
            trading_status.market_order_available_flag
            and trading_status.api_trade_available_flag
        ):
            amount = await self.get_position_quantity()
            if amount == 0:
                return
            await self.place_order(OrderDirection.ORDER_DIRECTION_SELL, quantity=amount // self.lot)
            logger.info("Sell position to zero. figi=%s amount=%s", self.figi, amount)

    async def shutdown(self):
        """
        Function on closing console
        """
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
        await self.sell_all()
        await self.main_cycle()