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
import yaml
from pathlib import Path
from torch.utils.data import ConcatDataset

#settings come from gz_foundation_models.yaml, or the file named by CONFIG (e.g. gz_foundation_models_radio.yaml)
config = yaml.safe_load(open(os.environ.get("CONFIG", Path(__file__).with_suffix(".yaml"))))
data_config = config["data"]

WEIGHTS_DIR = config["paths"]["weights_dir"]
REP_DIR = config["paths"]["rep_dir"]
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

if data_config["dataset"] == "radio":
    #RGZ FIRST for train/val; all of MiraBest (train + test) is the labelled "class" set used for accuracy
    train, val, mirabest_train, mirabest_test = gz.radio_galaxy_zoo_data_loaders(rgz_dir = data_config["rgz_dir"],
                                                                              mirabest_dir = data_config["mirabest_dir"],
                                                                              mirabest_subset = data_config["mirabest_subset"],
                                                                              train_split = data_config["train_split"],
                                                                              num_workers = data_config["num_workers"])
    class_ = DataLoader(ConcatDataset([mirabest_train.dataset, mirabest_test.dataset]), 128, num_workers = data_config["num_workers"])
else:
    train,val,class_  = gz.Galaxy_zoo_data_loaders(galaxyzoo_dir = data_config["galaxyzoo_dir"],
                                                   galaxyzooq_dir = data_config["galaxyzoo_class_dir"],
                                                   train_split = data_config["train_split"],
                                                   num_workers = data_config["num_workers"])


def evaluate_model(name, load_model):
    #every encoder gets train, val and class splits; splits saved by a previous run are skipped
    loaders = {"train": train, "val": val, "class": class_}
    missing = [split for split in loaders if not os.path.exists(rep_path(f"{name}_{split}"))]
    if not missing:
        print(f"[{name}] representations already exist in {REP_DIR}, skipping.")
        return

    model = load_model()
    for split in missing:
        print(f"[{name}] extracting {split} representations...")
        rep, label, rep_ids = Custom.get_representations(model=model, loader=loaders[split], encoder=True, labeled=(split == "class"))
        save_representations(f"{name}_{split}", rep, label, rep_ids)
    print(f"[{name}] done.")

    #free the encoder before loading the next one
    del model
    torch.cuda.empty_cache()


## Model: Zoobot ConvNext Base
evaluate_model("zoobot_convnext_base",
               lambda: timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_base', pretrained=True, num_classes=0))


## Model: Zoobot ConvNext Large
evaluate_model("zoobot_convnext_large",
               lambda: timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_large', pretrained=True, num_classes=0))


## Model: DINOv3 ConvNext Base
evaluate_model("dinov3_convnext_base",
               lambda: torch.hub.load("../dinov3", 'dinov3_convnext_base', source='local',
                                      weights=os.path.join(WEIGHTS_DIR, "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth")))


## Model: DINOv3 ConvNext Large
evaluate_model("dinov3_convnext_large",
               lambda: torch.hub.load("../dinov3", 'dinov3_convnext_large', source='local',
                                      weights=os.path.join(WEIGHTS_DIR, "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth")))


## Model: DINOv3 ViT-H/16+
evaluate_model("dinov3_vith16plus",
               lambda: torch.hub.load("../dinov3", 'dinov3_vith16plus', source='local',
                                      weights=os.path.join(WEIGHTS_DIR, "dinov3_vith16plus.pth")))


## Model: DINOv3 ViT-S/16
evaluate_model("dinov3_vits16",
               lambda: torch.hub.load("../dinov3", 'dinov3_vits16', source='local',
                                      weights=os.path.join(WEIGHTS_DIR, "dinov3_vits16.pth")))


## Model: ImageNet ConvNext Base
def load_imnet_convnext_base():
    model = tv.models.convnext_base(weights=tv.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Identity()
    return model

evaluate_model("imagenet_convnext_base", load_imnet_convnext_base)


## Model: ImageNet ConvNext Large
def load_imnet_convnext_large():
    model = tv.models.convnext_large(weights=tv.models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    model.classifier[-1] = torch.nn.Identity()
    return model

evaluate_model("imagenet_convnext_large", load_imnet_convnext_large)


## Model: ImageNet ResNet18
def load_imnet_resnet18():
    model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    return model

evaluate_model("imagenet_resnet18", load_imnet_resnet18)




