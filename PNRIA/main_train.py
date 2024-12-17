from PNRIA.configs.config import GlobalConfig, load_yaml
from PNRIA.torch_c.trainer import Trainer

<<<<<<< HEAD

config = load_yaml("configs/config_model_unet.yml")
cg = GlobalConfig(config)

t = Trainer.from_config(config)
=======
config = load_yaml("configs/config_model_unet.yml")
>>>>>>> modular-integration

GlobalConfig(config)
print(GlobalConfig().to_dict())
t = Trainer.from_config(config)
t.train()
