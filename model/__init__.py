from model.baselines import *
from model.wavenet import *


def get_model(model_name: str, 
              model_args: dict,
              **kwargs) -> Optional[LightningModule]:

    model_args.update(kwargs)

    if model_name == "mlp":
        model = MLP(**model_args)
        
    elif model_name == "lstm":
        model = LSTM(**model_args)
        
    elif model_name == "wavenet":
        model = WaveNet(**model_args)
        
    else:
        raise ValueError(f"{model_name} is not a valid model")
    
    return model
