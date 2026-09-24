import torch
import torchvision as tv
import torchvision.models as models
import backbone.data_handle.Test as test
import timm
import backbone.data_handle.Custom as Custom
from torch.utils.data import Dataset, DataLoader
import backbone.visuals.VISUAL as viz
import backbone.data_handle.GalaxyZoo as gz
import importlib
import backbone.custom_metrics.AstroMLmodified as AstroMLmod
import os
import pandas as pd

WEIGHTS_DIR = "data_models"
REP_DIR = "/idia/projects/camil/Koketso/galaxy_zoo_representations"
os.makedirs(REP_DIR, exist_ok=True)

def save_representations(name, rep, labels, ids):
    #persist one model's representation set as its own parquet file
    df = pd.DataFrame(rep)
    df["label"] = labels
    df["id"] = ids
    df.to_parquet(os.path.join(REP_DIR, f"{name}.parquet"))


#loading the data

train,val,class_  = gz.Galaxy_zoo_data_loaders(train_split = 0.8, num_workers = 30)


def evaluate_model(name, model, unlabeled_loader):
    #extract representations for one encoder and persist them as parquet
    print(f"[{name}] extracting unlabeled representations...")
    rep, label, rep_ids = Custom.get_representations(model=model, loader=unlabeled_loader, encoder=True, labeled=False)

    print(f"[{name}] extracting labeled representations...")
    rep_c, label_c, rep_ids_c = Custom.get_representations(model=model, loader=class_, encoder=True, labeled=True)

    print(f"[{name}] saving representations to parquet...")
    save_representations(f"{name}_val", rep, label, rep_ids)
    save_representations(f"{name}_class", rep_c, label_c, rep_ids_c)
    print(f"[{name}] done.")


## Model: Zoobot ConvNext Base
zoobot_encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_base', pretrained=True, num_classes=0)
evaluate_model("zoobot_convnext_base", zoobot_encoder, val)


## Model: Zoobot ConvNext Large
zoobot_large_encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_large', pretrained=True, num_classes=0)
evaluate_model("zoobot_convnext_large", zoobot_large_encoder, val)


## Model: DINOv3 ConvNext Base
dinov3_convnext_base = torch.hub.load("../dinov3", 'dinov3_convnext_base', source='local',
                                       weights=os.path.join(WEIGHTS_DIR, "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"))
evaluate_model("dinov3_convnext_base", dinov3_convnext_base, val)


## Model: DINOv3 ConvNext Large
dinov3_convnext_large = torch.hub.load("../dinov3", 'dinov3_convnext_large', source='local',
                                        weights=os.path.join(WEIGHTS_DIR, "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"))
evaluate_model("dinov3_convnext_large", dinov3_convnext_large, val)


## Model: DINOv3 ViT-H/16+
dinov3_vith = torch.hub.load("../dinov3", 'dinov3_vith16plus', source='local',
                              weights=os.path.join(WEIGHTS_DIR, "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"))
evaluate_model("dinov3_vith16plus", dinov3_vith, train)


## Model: DINOv3 ViT-S/16
dinov3_vits16 = torch.hub.load("../dinov3", 'dinov3_vits16', source='local',
                                weights=os.path.join(WEIGHTS_DIR, "dinov3_vits16.pth"))
evaluate_model("dinov3_vits16", dinov3_vits16, train)


## Model: ImageNet ConvNext Base
imnet_convnext_base = tv.models.convnext_base(weights=tv.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
imnet_convnext_base.classifier[-1] = torch.nn.Identity()
evaluate_model("imagenet_convnext_base", imnet_convnext_base, train)


## Model: ImageNet ConvNext Large
imnet_convnext_large = tv.models.convnext_large(weights=tv.models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
imnet_convnext_large.classifier[-1] = torch.nn.Identity()
evaluate_model("imagenet_convnext_large", imnet_convnext_large, train)


## Model: ImageNet ResNet18
imnet_resnet18 = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
imnet_resnet18.fc = torch.nn.Identity()
evaluate_model("imagenet_resnet18", imnet_resnet18, train)




