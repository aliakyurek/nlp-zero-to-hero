import os
import yaml

import torch
import pytorch_lightning as pl

def init_env(script_dir,yml_file):
    with open(os.path.join(script_dir, yml_file), 'r') as file:
        params = yaml.safe_load(file)
    pl.seed_everything(params['app']['manual_seed'])
    torch.cuda.empty_cache()
    return params

class PlApp:
    def __init__(self, params, data_module, model, cls_experiment, ckpt_monitor=None, ckpt_path=None):
        self.ckpt_path = ckpt_path
        self.data_module = data_module
        self.model = model
        self.params = params
        p=params['app']

        if self.ckpt_path:
            self.experiment = cls_experiment.load_from_checkpoint(self.ckpt_path, model=model,
                                                                 **self.params['experiment'])
        else:
            self.experiment = cls_experiment(model=model, **self.params['experiment'])

        logger = pl.loggers.TensorBoardLogger(save_dir=p['logs_dir'], name=p['name'], default_hp_metric=False,
                                              version=None)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=self.params['trainer']['monitor'], mode=self.params['trainer']['mode'],
            dirpath=os.path.join(p['logs_dir'],p['name']), save_top_k=1, save_weights_only=False,
            filename=f"version={logger.version:02d}-" + "{epoch:02d}-{"+self.params['trainer']['monitor']+":.3f}"
        )
        self.trainer = pl.Trainer(gpus=1, accelerator="gpu", max_epochs=self.params['trainer']['max_epochs'],
                                  benchmark=False, deterministic=True, num_sanity_val_steps=0,
                                  enable_model_summary=False, enable_progress_bar=True,
                                  logger=logger, callbacks=[checkpoint_callback],
                                  limit_train_batches=None, limit_val_batches=None, limit_test_batches=None,
                                  gradient_clip_val=1.0)

        self.trainer.logger.log_hyperparams(self.params)

    def train(self):
        return self.trainer.fit(self.experiment, datamodule=self.data_module)

