import csv
import os
from datetime import datetime

import asyncio
import logging

from typing import List, Optional

from tinkoff.invest import AioRequestError
from tinkoff.invest.async_services import AsyncServices
from tinkoff.invest.utils import quotation_to_decimal


logger = logging.getLogger(__name__)

class PortfolioLogger:
    def __init__(
        self,
        log_interval: int,
        client: Optional[AsyncServices],
        path_to_logs: str,
    ):
        self.account_id = None
        self.log_interval = log_interval
        self.client = client
        self.log_file = path_to_logs + 'portfolio_log.csv'

    def _initialize_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "balance", "portfolio_value", "total_amount_shares","expected_yield"])

    async def get_portfolio_data(self):
        portfolio = await self.client.operations.get_portfolio(account_id=self.account_id)
        balance = quotation_to_decimal(portfolio.total_amount_currencies)
        portfolio_value = quotation_to_decimal(portfolio.total_amount_portfolio)
        expected_yield = quotation_to_decimal(portfolio.expected_yield)
        total_amount_shares = quotation_to_decimal(portfolio.total_amount_shares)
        return balance, portfolio_value, expected_yield, total_amount_shares

    
    async def log_portfolio(self):
        balance, portfolio_value, expected_yield, total_amount_shares = await self.get_portfolio_data()
        
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), balance, portfolio_value, total_amount_shares, expected_yield])
    
    async def main_cycle(self):
        while True:
            await self.log_portfolio()
            await asyncio.sleep(self.log_interval)

    async def start(self):
        self._initialize_log()
        logger.info("Start logging portfolio.")
        if self.account_id is None:
            try:
                self.account_id = (
                    (await self.client.users.get_accounts()).accounts.pop().id
                )
            except AioRequestError as are:
                logger.error("Error taking account id. Stopping protfolio logging. %s", are)
                return
        await self.main_cycle()