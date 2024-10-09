from PNRIA.torch_c.trainer import Trainer

t = Trainer.from_config("configs/config_model_unet.yml")

t.train()