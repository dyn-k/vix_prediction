import unittest

from data.download import *
from data.preprocess import *
import torch


class TestData(unittest.TestCase):
    
    def test_dataload(self):
        # download init data
        self.assertTrue(isinstance(RawData.get('snp500'), pd.DataFrame))
        
        # after download - read from ./data/raw folder
        data = RawData.get('snp500')
        self.assertTrue(isinstance(data, pd.DataFrame))
        self.assertTrue('Close' in data.columns)
        self.assertTrue(len(data) > 0)
          
    def test_tsdataset(self):
        # test ts dataset instance type and size
        data = RawData.get("vix")
        dataset = TSDataset(data, seq_len=10, pred_len=5, target_col=0)
        self.assertTrue((dataset.__getitem__(0)[0].shape == torch.Size([10, len(data.columns)])))
        self.assertTrue((dataset.__getitem__(0)[1].shape == torch.Size([1, 1])))
    
    def test_tsloader(self):
        # test get_loaders after split_train_test
        features = Features(vol="vix")
        with self.assertRaises(ValueError):
            features.get_loaders(seq_len=10, pred_len=5, batch_size=8)
        
        # test get_loaders instance type and size
        train_loader, test_loader = features.split_train_test(ratio=0.8).get_loaders(seq_len=10, pred_len=5, batch_size=8)
        self.assertTrue(next(iter(train_loader))[0].shape == torch.Size([8, 10, 1]))
        self.assertTrue(next(iter(train_loader))[1].shape == torch.Size([8, 1, 1]))
    
    
if __name__ == '__main__':
    unittest.main()