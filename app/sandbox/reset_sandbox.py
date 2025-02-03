import logging
import os
import shutil

from decimal import Decimal
from dotenv import load_dotenv

from tinkoff.invest import MoneyValue, Client
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX 
from tinkoff.invest.utils import decimal_to_quotation

load_dotenv(override=True)
TOKEN = os.getenv("TOKEN")
IS_SANDBOX = True if os.getenv("SANDBOX") == 'True' else False

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)

def add_money_sandbox(client, account_id, money, currency="rub"):
    """
    Function to add money to sandbox account.
    """
    money = decimal_to_quotation(Decimal(money))
    return client.sandbox.sandbox_pay_in(
        account_id=account_id,
        amount=MoneyValue(units=money.units, nano=money.nano, currency=currency),
    )

def main():
    target = INVEST_GRPC_API_SANDBOX
    with Client(token=TOKEN, target=target) as client:
        sandbox_account = client.users.get_accounts().accounts[0]
        account_id = sandbox_account.id
        sandbox_accounts = client.users.get_accounts()
        print(sandbox_accounts)

        for sandbox_account in sandbox_accounts.accounts:
            client.sandbox.close_sandbox_account(account_id=sandbox_account.id)

        sandbox_account = client.sandbox.open_sandbox_account()

        account_id = sandbox_account.account_id

        print(add_money_sandbox(client=client, account_id=account_id, money=1000000))


if __name__ == "__main__":
    if os.path.exists('sandbox_logs'):
        shutil.rmtree('sandbox_logs')
    main()