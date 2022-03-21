from matplotlib import pyplot as plt
from pandas import read_parquet 
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np 
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel, ExponentialSmoothing
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf

from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, SunspotsDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries

import warnings

warnings.filterwarnings("ignore")
import logging


pd.set_option('display.max_rows', 500)
pd.set_option("display.max_columns", 100)


RES_PATH = 'res/'
df = pd.read_parquet(RES_PATH + 'df_pixel.parquet')
df.head()
df = df.set_index('date')

def prepare_input(df, start_col=0, end_col=1000):
    transformer = Scaler() 
    train, val = df.iloc[161:245, start_col:end_col], df.iloc[245:,start_col:end_col]
    train = TimeSeries.from_dataframe(train, freq='MS')
    val = TimeSeries.from_dataframe(val, freq='MS')

    train_ts = transformer.fit_transform(train)
    val_ts = transformer.transform(val)
    year_series = datetime_attribute_timeseries(
        pd.date_range(start=train_ts.start_time(), freq=train_ts.freq_str, periods=96),
        attribute="year",
        one_hot=False,
    )
    year_series = Scaler().fit_transform(year_series)
    month_series = datetime_attribute_timeseries(
        year_series, attribute="month", one_hot=True
    )
    covariates = year_series.stack(month_series)
    cov_train, cov_val = covariates.split_after(pd.Timestamp("20201201"))

    return (train_ts, val_ts), (cov_train, cov_val), covariates


def build_model(n_epochs=100):
    return RNNModel(
        model="GRU",
        hidden_dim=1500,
        dropout=0.1,
        batch_size=16,
        n_epochs=n_epochs,
        optimizer_kwargs={"lr": 1e-3},
        model_name="Air_RNN",
        log_tensorboard=False,
        random_state=42,
        training_length=12,
        input_chunk_length=10,
        force_reset=True,
        save_checkpoints=False,
    )
    
import darts 
import pickle 

def eval_model(model, val_ts, n):
        pred_series = model.predict(n=12, future_covariates=covariates)
        plt.figure(figsize=(8, 5))
        val_ts.plot(label="actual")
        pred_series.plot(label="forecast")
        plt.legend()
        plt.savefig(f'{n}.png')
        return darts.metrics.mape(pred_series, val_ts)

for start in range(13000, 16000, 1000):
    (train_ts, val_ts), (cov_train, cov_val), covariates = prepare_input(df, start, start+1000)
    
    model = build_model(200)

    model.fit(
        train_ts,
        future_covariates=covariates,
        verbose=True,
    )
    mape = eval_model(model, val_ts, n=start)
    print(mape)
    print(f'Start: {start}, End: {start+1000}')
    
    print(f'MAPE: {mape}')
    with open(f'model_{start}.pkl', 'wb') as f:
        pickle.dump(model, f)