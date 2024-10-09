from PNRIA.torch_c import trainer

t = trainer.from_config("configs/config_model_unet.yml")

t.train()