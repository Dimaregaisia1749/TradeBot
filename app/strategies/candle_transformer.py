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
from tinkoff.invest.utils import now

from datetime import timedelta
from pathlib import Path

from torch import nn
import torch

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
            nn.Linear(32, 5)
            )

    def forward(self, x):
        """
        # prices: [B, 180, 5] (OHLCV)
        # indicators: [B, 180, 4]
        # time_feats: [B, 180, 3]
        """
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
        model_dir = 'checkpoints/'
        self.MODEL_PATH = os.path.join(model_dir, 'best.tar')
        self.model = self.__init_model()
        logger.info("Start TransformerStrategy. figi=%s", self.figi)

    def __init_model(self):
        heads = 4
        encoder_layers = 3
        d_model = 128
        window_size = 240
        num_workers = 4
        model = CandleTransformer(
            heads=heads,
            encoder_layers=encoder_layers, 
            d_model=d_model,
            ).to(device=device)
        checkpoint = torch.load(self.MODEL_PATH, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    async def get_data(self):
        async for candle in self.client.get_all_candles(
            figi=self.figi,
            from_=now() - timedelta(hours=3),
            to=now(),
            interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
        ):
            print(candle.time, candle.is_complete)

    async def trade(self):
        """
        Decision maker.
        """
        asyncio.gather(self.get_data())