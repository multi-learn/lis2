from PNRIA.configs.config import GlobalConfig, load_yaml
from PNRIA.torch_c.trainer import Trainer

config = load_yaml("configs/config_model_unet.yml")

GlobalConfig(config)
print(GlobalConfig().to_dict())
t = Trainer.from_config(config)
print(t.train())
