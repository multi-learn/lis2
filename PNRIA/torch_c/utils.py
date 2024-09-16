import torch

def position_encoding(positions, mode):
    if mode == "linear":
        return torch.tensor(positions / 114000)
    elif mode == "sym":
        positions[:,:,0] = torch.absolute(torch.sin(((57000 - positions[:,:,0]) * 0.00319444444400 + 180) * torch.pi / 360))
        positions[:,:,1] = torch.absolute(torch.sin(((900 - positions[:,:,1]) * 0.00319444444400) * torch.pi / 360))
        return positions.permute(0, 3, 1, 2) # position encoding