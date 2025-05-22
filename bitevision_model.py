import torch
import torchvision

def create_effnetb2(num_classes:int=101, seed:int=42, device:str="cpu"):
    
    weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT
    effnetb2=torchvision.models.efficientnet_b2(weights=weights).to(device)
    
    for params in effnetb2.parameters():
        params.requires_grad=False
        
    torch.manual_seed(seed)
    effnetb2.classifier=torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features=1408, out_features=101)
        )
    
    effnetb2_transforms=weights.transforms()
    
    return effnetb2, effnetb2_transforms


