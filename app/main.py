import json
import logging
import os

from dotenv import load_dotenv

from app.trade_agent import TradeAgent


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

if __name__ == "__main__":
    with open('instruments_config.json') as f:
        data = json.load(f)
    portfolio = list(data["instruments"].values())

    trade_agent = TradeAgent(token=TOKEN, is_sandbox=IS_SANDBOX, porfolio=portfolio)
    trade_agent.start()
    