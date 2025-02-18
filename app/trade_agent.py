
import asyncio
import logging
import signal
import sys

from tinkoff.invest import AsyncClient, CandleInterval, InstrumentIdType
from tinkoff.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX 
from tinkoff.invest.utils import now

from app.strategies.strategy_solver import resolve_strategy
from app.portfolio_logger import PortfolioLogger


logger = logging.getLogger(__name__)

class TradeAgent:
    """
    Class for controlling strategies.
    """
    def __init__(
        self,
        token: str,
        is_sandbox: bool,
        porfolio: list,
        path_to_logs: str,
    ):
        self.portfolio = porfolio
        self.TOKEN = token
        self.IS_SANDBOX = is_sandbox
        self.PATH_TO_LOGS = path_to_logs
        self.strategies = []
        self.portfolio_logger = None
    
    def handle_signal(self):
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        logger.info("Shutdown agent by signal")
        shutdown_tasks = []
        for strategy in self.strategies:
            shutdown_tasks.append(strategy.shutdown())
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        asyncio.get_running_loop().stop()
        sys.exit(1)

    async def run_strategies(self, portfolio, timeframe, days_back, check_interval):
        """
        From this function starts
        strategy for every ticker from portfolio.
        """
        target = INVEST_GRPC_API_SANDBOX if self.IS_SANDBOX else INVEST_GRPC_API 
        if self.IS_SANDBOX:
            logger.info("Start trade agent on sandbox account.")
        else:
            logger.info("Start trade agent on prod account.")
        async with AsyncClient(token=self.TOKEN, app_name="TinkoffApp", target=target) as client:
            self.portfolio_logger = PortfolioLogger(log_interval=60, client=client, path_to_logs=self.PATH_TO_LOGS)
            portfolio_logger_task = asyncio.create_task(self.portfolio_logger.start())
            strategy_tasks = []
            for instrument in portfolio:
                strategy = instrument["strategy"]
                figi = instrument["figi"]
                lot = await self.get_lot_size(figi=figi, client=client)
                strategy = resolve_strategy(
                    strategy_name=strategy,
                    figi=figi,
                    timeframe=timeframe,
                    check_interval=check_interval,
                    days_back=days_back,
                    lot=lot,
                    client=client
                )
                self.strategies.append(strategy)
                strategy_tasks.append(asyncio.create_task(strategy.start()))
            await asyncio.gather(*strategy_tasks, portfolio_logger_task)

    async def get_lot_size(self, figi: str, client):
        response = await client.instruments.get_instrument_by(
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
            id=figi
        )
        return response.instrument.lot

    def start(self):
        timeframe = CandleInterval.CANDLE_INTERVAL_5_MIN
        days_back = 1
        check_interval = 60*5 
        loop = asyncio.get_event_loop()
        signal.signal(signal.SIGINT, lambda s, f: self.handle_signal()) #close console handler
        signal.signal(signal.SIGBREAK, lambda s, f: self.handle_signal())
        task = loop.create_task(
            self.run_strategies(
                portfolio=self.portfolio,
                timeframe=timeframe,
                days_back=days_back,
                check_interval=check_interval,
            )
        )
        loop.run_until_complete(task)
        