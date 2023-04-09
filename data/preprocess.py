import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import torch
from torch.utils.data import Dataset, DataLoader

from data.download import RawData
from utils.types import *


class Features:
    """
    Generate input features for training.
    This returns two types of features)
    1. pd.DataFrame - for statistical models
    2. torch.DataLoader - for torch(deep learning) models
    """
    def __init__(self, 
                 vol: str, 
                 under_asset: str="",
                 ) -> None:
        
        self.file_name = vol + "_" + under_asset if under_asset else vol
        self.vol_name = vol
        self.under_asset_name = under_asset
        self.under_asset = None
        self.train, self.test = None, None
        self.is_arima = ''
        
        if not os.path.exists(f"./data/features/{self.file_name}.csv"):
            data = []

            vix = RawData.get(self.vol_name)['Close'].to_frame().rename(columns={'Close': self.vol_name})
            data.append(vix)
            
            if under_asset: 
                self.under_asset = RawData.get(self.under_asset_name)['Close'].to_frame()
                self.under_asset = self._log_return(self.under_asset)
                data.append(self.under_asset.rename(columns={'Close': self.under_asset_name}))
                
            self.data = pd.concat(data, axis=1)
            self.data.to_csv(f"./data/features/{self.file_name}.csv", index=True, encoding="utf-8")
            
        else:
            self.data = pd.read_csv(f"./data/features/{self.file_name}.csv", index_col="Date", encoding="utf-8")
            if under_asset:
                self.under_asset = self.data[under_asset].to_frame()
            
        
    def __repr__(self) -> DataFrame:
        if not isinstance(self.train, pd.DataFrame):
            return self.data.__repr__()
        else:
            return self.train.__repr__() + "\n" + self.test.__repr__()
        
    @property
    def columns(self) -> List[str]:
        if not isinstance(self.train, pd.DataFrame):
            return list(self.data.columns)
        else:
            return list(self.train.columns)
    
    def _add_arima(self, data: DataFrame, is_test: bool) -> DataFrame:
        # ARIMA of independant variable - underlying asset
        if is_test:
            history = list(self.train.iloc[:, 1].values)
            preds = []
            for i in tqdm(range(len(data)), desc="ARIMA fit"):
                model = ARIMA(order=self.arimaperiods,
                              endog=history).fit()
                preds.append(model.forecast()[0])
                history.append(data.iloc[i, 1])
            preds = pd.DataFrame({'predicted_mean': preds})
            preds.index = data.index
                
        else:
            model = ARIMA(order=self.arimaperiods,
                               endog=data.iloc[:, 1]).fit()
            preds = model.predict().to_frame()
            
        return pd.concat([data, preds], axis=1)
    
    @staticmethod
    def _log_return(data: Series) -> Series:
        data = np.abs(np.log(data).diff(1))
        data.fillna(0, inplace=True)
        return data
    
    def add_features(self, arimaperiods: Optional[List[int]]=None):

        self.file_name = self.file_name if not arimaperiods else\
                         self.file_name + "_arima"
        self.is_arima = self.is_arima if not arimaperiods else "_arima"

        def _add_features(data: DataFrame, postfix: str) -> DataFrame:
            file_name = f"./data/features/{self.file_name}_{postfix}.csv"
            if os.path.exists(file_name):
                data = pd.read_csv(file_name, index_col="Date", encoding="utf-8")

            else:
                if arimaperiods:
                    self.arimaperiods = arimaperiods
                    if not isinstance(self.under_asset, pd.DataFrame):
                        raise ValueError("ARIMA feature needs under_asset feature. e.g. 'snp500'")
                    
                    data = self._add_arima(data, postfix=="test")
                
                data.to_csv(file_name, index=True, encoding="utf-8")
            return data

        self.train = _add_features(self.train, "train")
        self.test = _add_features(self.test, "test")
        return self
        
    def split_train_test(self, ratio: float=0.9):
        split_index = int(len(self.data)*ratio)
        self.train = self.data.iloc[:split_index, :]
        self.test = self.data.iloc[split_index:, :]
        return self
    
    def get_loaders(self,
                    seq_len:int=20, 
                    pred_len:int=10,
                    batch_size:int=16
                    ) -> List[TorchDataLoader]:
        """
        - x: (batch_size, seq_len, features)
        - y: (batch_size, 1, 1)
        """
        
        if not isinstance(self.train, pd.DataFrame):
            raise ValueError("Train/test data are not generated. Call split_train_test() first.")
        
        dataloaders = []
        for shuffle, data in zip([True, False],[self.train, self.test]):
            dataset = TSDataset(data, seq_len=seq_len, pred_len=pred_len, target_col=0)
            dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
                       
        return dataloaders


class TSDataset(Dataset):
    """
    Customization for time-series dataset
    """
    def __init__(self, 
                 tsdata: DataFrame, 
                 seq_len: int, 
                 pred_len: int,
                 target_col: int) -> None:
        super().__init__()
        self.tsdata = tsdata
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target = target_col
        
    def __len__(self) -> int:
        return len(self.tsdata) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, ind: int) -> Tuple[TorchTensor, TorchTensor]:
        """
        x: (seq_len, features)
        y: (1, 1)
        """
        x = torch.tensor(self.tsdata.iloc[ind: ind + self.seq_len, :].to_numpy(),
                         dtype=torch.float32)
        y = torch.tensor(self.tsdata.iloc[ind + self.seq_len + self.pred_len - 1, self.target], 
                         dtype=torch.float32).reshape(-1, 1)
        return x, y