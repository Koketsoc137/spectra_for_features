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

def rep_path(name):
    return os.path.join(REP_DIR, f"{name}.parquet")


def save_representations(name, rep, labels, ids):
    #persist one model's representation set as its own parquet file
    df = pd.DataFrame(rep)
    df["label"] = labels
    df["id"] = ids
    #write to a temp file first so an interrupted job never leaves a partial parquet behind
    tmp_path = rep_path(name) + ".tmp"
    df.to_parquet(tmp_path)
    os.replace(tmp_path, rep_path(name))


#loading the data

train,val,class_  = gz.Galaxy_zoo_data_loaders(train_split = 0.8, num_workers = 30)


def evaluate_model(name, load_model, unlabeled_loader):
    #skip encoders whose representations were already saved by a previous run
    if all(os.path.exists(rep_path(f"{name}_{split}")) for split in ("val", "class")):
        print(f"[{name}] representations already exist in {REP_DIR}, skipping.")
        return

    #extract representations for one encoder and persist them as parquet
    model = load_model()
    print(f"[{name}] extracting unlabeled representations...")
    rep, label, rep_ids = Custom.get_representations(model=model, loader=unlabeled_loader, encoder=True, labeled=False)

    print(f"[{name}] extracting labeled representations...")
    rep_c, label_c, rep_ids_c = Custom.get_representations(model=model, loader=class_, encoder=True, labeled=True)

    print(f"[{name}] saving representations to parquet...")
    save_representations(f"{name}_val", rep, label, rep_ids)
    save_representations(f"{name}_class", rep_c, label_c, rep_ids_c)
    print(f"[{name}] done.")

    #free the encoder before loading the next one
    del model
    torch.cuda.empty_cache()


## Model: Zoobot ConvNext Base
evaluate_model("zoobot_convnext_base",
               lambda: timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_base', pretrained=True, num_classes=0),
               val)


## Model: Zoobot ConvNext Large
evaluate_model("zoobot_convnext_large",
               lambda: timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_large', pretrained=True, num_classes=0),
               val)


## Model: DINOv3 ConvNext Base
evaluate_model("dinov3_convnext_base",
               lambda: torch.hub.load("../dinov3", 'dinov3_convnext_base', source='local',
                                      weights=os.path.join(WEIGHTS_DIR, "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth")),
               val)


## Model: DINOv3 ConvNext Large
evaluate_model("dinov3_convnext_large",
               lambda: torch.hub.load("../dinov3", 'dinov3_convnext_large', source='local',
                                      weights=os.path.join(WEIGHTS_DIR, "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth")),
               val)


## Model: DINOv3 ViT-H/16+
evaluate_model("dinov3_vith16plus",
               lambda: torch.hub.load("../dinov3", 'dinov3_vith16plus', source='local',
                                      weights=os.path.join(WEIGHTS_DIR, "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth")),
               train)


## Model: DINOv3 ViT-S/16
evaluate_model("dinov3_vits16",
               lambda: torch.hub.load("../dinov3", 'dinov3_vits16', source='local',
                                      weights=os.path.join(WEIGHTS_DIR, "dinov3_vits16.pth")),
               train)


## Model: ImageNet ConvNext Base
def load_imnet_convnext_base():
    model = tv.models.convnext_base(weights=tv.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Identity()
    return model

evaluate_model("imagenet_convnext_base", load_imnet_convnext_base, train)


## Model: ImageNet ConvNext Large
def load_imnet_convnext_large():
    model = tv.models.convnext_large(weights=tv.models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Identity()
    return model

evaluate_model("imagenet_convnext_large", load_imnet_convnext_large, train)


## Model: ImageNet ResNet18
def load_imnet_resnet18():
    model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    return model

evaluate_model("imagenet_resnet18", load_imnet_resnet18, train)




