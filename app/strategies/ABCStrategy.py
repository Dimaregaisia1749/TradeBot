from app.strategies.base import BaseStrategy

import asyncio
import logging

from tinkoff.invest import OrderDirection

logger = logging.getLogger(__name__)

class ABCStrategy(BaseStrategy):
    async def trade(self):
        """
        Decision maker.
        """
        logger.info("Start ABC. figi=%s", self.figi)

        prev_price = await self.get_price()
        while True:
            await asyncio.sleep(60)
            curr_price = await self.get_price()
            logger.info("Price. figi=%s price=%s", self.figi, curr_price)
            if curr_price > prev_price * 1.001:
                await self.place_order(OrderDirection.ORDER_DIRECTION_BUY, quantity=1)
            elif curr_price < prev_price * 0.999:
                await self.place_order(OrderDirection.ORDER_DIRECTION_SELL, quantity=1)

            prev_price = curr_price