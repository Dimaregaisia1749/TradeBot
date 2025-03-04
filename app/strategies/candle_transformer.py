from app.strategies.base import BaseStrategy

import asyncio
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional

from tinkoff.invest import OrderDirection
from tinkoff.invest.async_services import AsyncServices
from tinkoff.invest import CandleInterval
from tinkoff.invest.caching.market_data_cache.cache import MarketDataCache
from tinkoff.invest.caching.market_data_cache.cache_settings import (
    MarketDataCacheSettings,
)
from tinkoff.invest.utils import now, quotation_to_decimal

from datetime import timedelta
from pathlib import Path

from torch import nn
import torch

from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

class TimeEncoder(nn.Module):
    def __init__(self, candles_features: int, time_features: int, d_model: int):
        super().__init__()
        self.input_proj = nn.Linear(candles_features, d_model)
        self.time2vec = nn.Sequential(
            nn.Linear(time_features, 32),
            nn.GELU(),
            nn.Linear(32, d_model)
        )
        self.learnable_pe = nn.Parameter(torch.randn(1, 5000, d_model))
        
    def forward(self, x):
        x, time_features = x[:, :, :5], x[:, :, 5:]
        t_emb = self.time2vec(time_features)
        x = self.input_proj(x)
        x = x + self.learnable_pe[:, :x.size(1), :] + t_emb
        return x

class Transformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, encoder_layers: int):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=encoder_layers
        )

    def forward(self, x):
        x = self.encoder(x).mean(dim=1)
        return x

class CandleTransformer(nn.Module):
    def __init__(self, heads: int, encoder_layers: int, d_model: int):
        super().__init__()
        self.time_enc = TimeEncoder(candles_features=5, time_features=3, d_model=d_model)
        self.transformer = Transformer(d_model=d_model, nhead=heads, encoder_layers=encoder_layers)
        self.out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.time_enc(x)
        trans_out = self.transformer(x)
        out = self.out(trans_out)
        return out

class TransformerStrategy(BaseStrategy):
    """
        Strategy on predicted candles from model
    """
    def __init__(
        self,
        figi: str,
        timeframe: CandleInterval,
        check_interval: int,
        lot: int,
        client: Optional[AsyncServices],
    ):
        self.account_id = None
        self.figi = figi
        self.timeframe = timeframe
        self.lot = lot
        self.check_interval = check_interval
        self.client = client
        model_dir = 'checkpoints/'
        self.MODEL_PATH = os.path.join(model_dir, 'best.tar')
        self.model = self.__init_model()
        self.window_size = 5
        logger.info("Start TransformerStrategy. figi=%s", self.figi)

        self.pred_dir = []
        self.actual_dir = []
        self.total_profit = 0

    def __init_model(self):
        heads = 4
        encoder_layers = 4
        d_model = 256
        model = CandleTransformer(
            heads=heads,
            encoder_layers=encoder_layers, 
            d_model=d_model,
            ).to(device=device)
        checkpoint = torch.load(self.MODEL_PATH, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def __form_df(self, candles):
        candles_list = [
            {
                'utc': candle.time,
                'open': float(quotation_to_decimal(candle.open)),
                'close': float(quotation_to_decimal(candle.close)),
                'high': float(quotation_to_decimal(candle.high)),
                'low': float(quotation_to_decimal(candle.low)),
                'volume': candle.volume,
            }
            for candle in candles
        ]
        df = pd.DataFrame(candles_list, columns=['utc', 'open', 'close', 'high', 'low', 'volume'])
        df['utc'] = pd.to_datetime(df['utc'], utc=True)
        df = df.set_index('utc')
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.day_of_week
        df['minute'] = df.index.minute
        return df

    def __normalize(self, tensor):
        mean = tensor[:, :-3].mean(dim=0, keepdim=True)
        std = tensor[:, :-3].std(dim=0, keepdim=True)
        epsilon = 1e-7
        normalized_data = (tensor[:, :-3] - mean) / (std + epsilon)
        normalized_data = torch.cat([normalized_data, tensor[:, -3:]], dim=-1)
        normalized_data[..., -3] = normalized_data[..., -3] / 23
        normalized_data[..., -2] = normalized_data[..., -2] / 6 
        normalized_data[..., -1] = normalized_data[..., -1] / 59
        return normalized_data, std, mean

    async def get_data(self):
        candles = []
        async for candle in self.client.get_all_candles(
            figi=self.figi,
            from_=now() - timedelta(seconds=self.window_size*self.check_interval),
            to=now(),
            interval=self.timeframe
        ):
            candles.append(candle)
        df = self.__form_df(candles=candles)
        tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
        tensor, std, mean = self.__normalize(tensor[:-1, :])
        tensor = tensor.unsqueeze(dim=0)
        return tensor, std, mean


    async def trade(self):
        """
        Decision maker.
        """
        quantity = 20
        input, std, mean = await self.get_data()
        self.model.eval()
        output = self.model(input)
        output = output.squeeze(dim=0)
        self.actual_dir.append(((input.squeeze(dim=0)[0, :][1] - input.squeeze(dim=0)[0, :][0]) >= 0.5).float().cpu())
        self.pred_dir.append((output >= 0.5).float().cpu())
        if output >= 0.5:
            buy_order = await self.place_order(OrderDirection.ORDER_DIRECTION_BUY, quantity=quantity)
            seconds_to_wait = self.check_interval - now().second - 1
            await asyncio.sleep(seconds_to_wait)
            sell_order = await self.place_order(OrderDirection.ORDER_DIRECTION_SELL, quantity=quantity)
            buy_price = float(quotation_to_decimal(buy_order.executed_order_price)) * quantity
            sell_price = float(quotation_to_decimal(sell_order.executed_order_price)) * quantity
            buy_comission = float(quotation_to_decimal(buy_order.executed_commission))
            sell_comission = float(quotation_to_decimal(sell_order.executed_commission))
            profit = sell_price - buy_price
            profit_with_comission =  profit - buy_comission - sell_comission
            self.total_profit += profit_with_comission
            print(f'Figi: {self.figi}. Buy for {buy_price}, Sell for {sell_price}, profit: {profit}, profit with commison" {profit_with_comission}')
        print(f'Figi: {self.figi} Total profit: {self.total_profit}, accuracy: {accuracy_score(self.actual_dir, self.pred_dir)}')
        
        