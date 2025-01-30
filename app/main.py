import json
import logging
import os

from dotenv import load_dotenv

from app.trade_agent import TradeAgent


load_dotenv(override=True)
TOKEN = os.getenv("TOKEN")
IS_SANDBOX = True if os.getenv("SANDBOX") == 'True' else False

if IS_SANDBOX:
    PATH_TO_LOGS = 'sandbox_logs/'
else:
    PATH_TO_LOGS = 'logs/'

os.makedirs(PATH_TO_LOGS, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(PATH_TO_LOGS + "base_log.log"),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    with open('instruments_config.json') as f:
        data = json.load(f)
    portfolio = list(data["instruments"].values())

    trade_agent = TradeAgent(token=TOKEN, is_sandbox=IS_SANDBOX, porfolio=portfolio, path_to_logs=PATH_TO_LOGS)
    trade_agent.start()
    