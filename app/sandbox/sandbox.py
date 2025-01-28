import logging
import os
from datetime import datetime
from decimal import Decimal
from dotenv import load_dotenv

from tinkoff.invest import MoneyValue
from tinkoff.invest.sandbox.client import SandboxClient
from tinkoff.invest.utils import decimal_to_quotation, quotation_to_decimal

load_dotenv()
TOKEN = os.getenv("TOKEN")

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def add_money_sandbox(client, account_id, money, currency="rub"):
    """Function to add money to sandbox account."""
    money = decimal_to_quotation(Decimal(money))
    return client.sandbox.sandbox_pay_in(
        account_id=account_id,
        amount=MoneyValue(units=money.units, nano=money.nano, currency=currency),
    )


def main():
    """Example - How to set/get balance for sandbox account.
    How to get/close all sandbox accounts.
    How to open new sandbox account.
    """
    with SandboxClient(TOKEN) as client:
        sandbox_account = client.users.get_accounts().accounts[0]
        print(sandbox_account)
        account_id = sandbox_account.id
        logger.info("orders: %s", client.orders.get_orders(account_id=account_id))
        logger.info(
            "positions: %s", client.operations.get_positions(account_id=account_id)
        )
        logger.info(
            "operations: %s",
            client.operations.get_operations(
                account_id=account_id,
                from_=datetime(2023, 1, 1),
                to=datetime(2023, 2, 5),
            ),
        )
        logger.info(
            "withdraw_limits: %s",
            client.operations.get_withdraw_limits(account_id=account_id),
        )
        for pos in client.operations.get_portfolio(account_id=account_id).positions:
            logger.info(
                "Pos %s: %s Amount",
                pos.figi,
                int(quotation_to_decimal(pos.quantity))
            )
        for pos in client.operations.get_portfolio(account_id=account_id).positions:
            logger.info(
                "Pos %s: %s RUB",
                pos.figi,
                round(quotation_to_decimal(pos.current_price)*quotation_to_decimal(pos.quantity))
            )
        logger.info(
            "Currencies: %s RUB",
            round(quotation_to_decimal(client.operations.get_portfolio(account_id=account_id).total_amount_currencies))
        )
        logger.info(
            "Total amount: %s RUB",
            round(quotation_to_decimal(client.operations.get_portfolio(account_id=account_id).total_amount_portfolio))
        )


if __name__ == "__main__":
    main()