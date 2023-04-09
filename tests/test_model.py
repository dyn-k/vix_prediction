import json
import unittest

from main import *

class test_model(unittest.TestCase):
    
    @staticmethod
    def _args_setup(model_name):
        json_text = Path(f'./configs/config_{model_name}.json').read_text()
        args = argparse.Namespace(**json.loads(json_text))
        args.model_name = model_name
        return args

    def test_mlp(self):
        model_name = "MLP".lower()
        args = self._args_setup(model_name)
        n_features = 3
        model = get_model(model_name, args.model, seq_len=args.data["seq_len"], n_features=n_features)
        self.assertEqual(model.__class__, MLP)
        
        data = torch.randn(args.data["batch_size"], args.data["seq_len"], n_features, dtype=torch.float32)
        out = model(data)
        self.assertEqual(out.shape, (args.data["batch_size"], 1, 1))
    
    def test_lstm(self):
        model_name = "LSTM".lower()
        args = self._args_setup(model_name)
        n_features = 3
        model = get_model(model_name, args.model, seq_len=args.data["seq_len"], n_features=n_features)
        self.assertEqual(model.__class__, LSTM)
        
        data = torch.randn(args.data["batch_size"], args.data["seq_len"], n_features, dtype=torch.float32)
        out = model(data)
        self.assertEqual(out.shape, (args.data["batch_size"], 1, 1))
    
    def test_main(self):
        model_name = "WaveNet".lower()
        args = self._args_setup(model_name)
        n_features = 3
        model = get_model(model_name, args.model, seq_len=args.data["seq_len"], n_features=n_features)
        self.assertEqual(model.__class__, WaveNet)
        
        data = torch.randn(args.data["batch_size"], args.data["seq_len"], n_features, dtype=torch.float32)
        out = model(data)
        self.assertEqual(out.shape, (args.data["batch_size"], 1, 1))
    