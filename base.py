import os
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.utilities import rank_zero_only
import re

def init_env(yml_file):
    with open(yml_file,'r') as file:
        params = yaml.safe_load(file)
    pl.seed_everything(params['app']['manual_seed'])
    torch.cuda.empty_cache()
    return params

class TBLogger(loggers.TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)

class PlApp:
    def __init__(self, params, data_module, model, cls_experiment, ckpt_path=None):
        self.ckpt_path = ckpt_path
        self.data_module = data_module
        self.model = model
        self.params = params
        p=params['app']
        version=None

        if self.ckpt_path:
            self.experiment = cls_experiment.load_from_checkpoint(self.ckpt_path, model=model,
                                                                 **self.params['experiment'])
            version = int(re.search(r"version=(.*?)-",self.ckpt_path).group(1))
        else:
            self.experiment = cls_experiment(model=model, **self.params['experiment'])

        logger = TBLogger(save_dir=p['logs_dir'], name=p['name'], default_hp_metric=False,
                                              version=version)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=self.params['trainer']['monitor'], mode=self.params['trainer']['mode'],
            dirpath=os.path.join(p['logs_dir'],p['name']), save_top_k=1, save_weights_only=False,
            filename=f"version={logger.version:02d}-" + "{epoch:02d}-{"+self.params['trainer']['monitor']+":.3f}"
        )
        self.trainer = pl.Trainer(accelerator="gpu", max_epochs=self.params['trainer']['max_epochs'],
                                  benchmark=False, deterministic=True, num_sanity_val_steps=0,
                                  enable_model_summary=False, enable_progress_bar=True,
                                  logger=logger, callbacks=[checkpoint_callback],
                                  limit_train_batches=None, limit_val_batches=None, limit_test_batches=None,
                                  gradient_clip_val=1.0)

        self.trainer.logger.log_hyperparams(self.params)

    def train(self):
        # Even though we load the checkpoint in constructor, if it's loaded to continue training,
        # load_from_checkpoint only restores model weights. However, fit restores model weights (a little double work but it's OK),
        # current epoch, current global step, optimizer states etc.
        return self.trainer.fit(self.experiment, datamodule=self.data_module, ckpt_path=self.ckpt_path)

