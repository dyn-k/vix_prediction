import os
import warnings
import sys
import json
import argparse
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.preprocess import Features
from model import *
from utils import *

seed_everything(0)
warnings.simplefilter("ignore")


def main(args):
    # extract data & model arguments
    data_args = args.data
    model_args = args.model

    # get features
    features = Features(
                    vol=data_args["vol"], 
                    under_asset=data_args["under_asset"]
               ).split_train_test(ratio=data_args["train_test_ratio"]
               ).add_features(arimaperiods=data_args["arimaperiods"])
    
    train_loader, test_loader = features.get_loaders(seq_len=data_args["seq_len"], 
                                                     pred_len=data_args["pred_len"], 
                                                     batch_size=data_args["batch_size"])

    # setup exp_name and log_dir
    exp_name = f"{args.model_name}{features.is_arima}"
    log_dir = f"./logs/{exp_name}/"

    model = get_model(model_name=args.model_name,
                      model_args=model_args, 
                      n_features=len(features.columns),
                      seq_len=data_args["seq_len"])

    # experiments starts
    os.makedirs(log_dir, exist_ok=True)
    if not os.listdir(log_dir):
        # set model checkpoint and logger
        ckpt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        filename="model", 
                                        save_top_k=1, 
                                        monitor="train_loss")
        
        logger = TensorBoardLogger(save_dir="./logs/",
                                   name=exp_name)
        
        # define trainer
        trainer = Trainer(logger=logger,
                          callbacks=[ckpt_callback],
                          max_epochs=100,
                          deterministic=True)
        
        # fit if empty
        trainer.fit(model, train_loader)
    
    else:
        trainer = Trainer()
        
    # model tests
    errors = trainer.test(
        model, test_loader, ckpt_path=log_dir+"model.ckpt", verbose=True
        )[0]
    
    preds = torch.concat(
        trainer.predict(model, test_loader, 
                        ckpt_path=log_dir+"model.ckpt")
        ).reshape(-1)
        
    # save - errors
    os.makedirs('./results/errors/', exist_ok=True)
    with open(f"./results/errors/{exp_name}.json", "w") as f:
        json.dump(errors, f, indent=4)
    
    # save - predictions
    os.makedirs('./results/predictions/', exist_ok=True)
    with open(f"./results/predictions/{exp_name}.csv", "w") as f:
        for _, val in enumerate(preds):
            f.write(f"{val}\n")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='wavenet', help='mlp/lstm/wavenet')
    parser.add_argument('--arima', type=bool, default=False, help='set  for arima feature')
    parser_args = parser.parse_args()
    
    json_text = Path(f'./configs/config_{parser_args.model}.json').read_text()
    args = argparse.Namespace(**json.loads(json_text))
    args.model_name = parser_args.model
    args.data['arimaperiods'] = [1,1,0] if parser_args.arima else []
    
    main(args)