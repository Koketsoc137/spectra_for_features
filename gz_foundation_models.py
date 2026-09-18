import torch
import torchvision.models as models
import backbone.Test as test
import timm
import backbone.Custom as Custom
from torch.utils.data import Dataset, DataLoader
import backbone.VISUAL as viz
import backbone.GalaxyZoo as gz
import importlib
import backbone.AstroMLmodified as AstroMLmod


#loading the data

train,val,class_  = gz.Galaxy_zoo_data_loaders(train_split = 0.8, num_workers = 30)


knns = []
tpcs = []
cas = []
ids = []




## Model: Zoobot ConvNext Base


zoobot_encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_base', pretrained=True, 
                                   num_classes=0)


zoobot_rep, zoobot_label, zoobot_ids = Custom.get_representations(model  = zoobot_encoder,
                                       loader = val,
                                       encoder = True,
                                       labeled = False)



zoobot_rep_c, zoobot_label_c, zoobot_ids_c = Custom.get_representations(model  = zoobot_encoder,
                                                                       loader = class_,
                                                                       encoder = True,
                                                                       labeled = True)


#umap_cs  =viz.umap(zoobot_rep_c)

pca_var,pca_dim = viz.pca(zoobot_rep,
                          return_variance_dimension = True)
knn = test.KNN_accuracy(zoobot_rep_c,
                        zoobot_label_c)
print("KNN: ",knn)
zc_accuracy = test.clustering_accuracy((zoobot_rep,zoobot_ids),(zoobot_label_c,zoobot_ids_c))
zoobot_tpcf = AstroMLmod.TPCF_score(zoobot_rep)
print("TPCF: ",zoobot_tpcf)
zoobot_id = AstroMLmod.id_score(zoobot_rep)
print("ID: ",zoobot_id)

knns.append(knn)
tpcfs.append(zoobot_tpcf)
cas.append(zc_accuracy)
ids.append(zoobot_id)


#dino convxnext base

dinov3_convnext_base = torch.hub.load("../dinov3", 'dinov3_convnext_base',
                                      source='local',
                                      weights="dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth")
dino_rep, dino_label, dino_ids = Custom.get_representations(model  = dinov3_convnext_base,
                           loader = val,
                           encoder = True,
                           labeled = False)

dino_rep_c, dino_label_c, dino_ids_c = Custom.get_representations(model  = dinov3_convnext_base,
                           loader = class_,
                           encoder = True,
                           labeled = True)

#umap_  =viz.umap(dino_rep)

dino_pca_var,dino_pca_dim = viz.pca(dino_rep,return_variance_dimension = True)
dino_knn = test.KNN_accuracy(dino_rep_c, dino_label_c)
print(dino_knn)
dc_accuracy = test.clustering_accuracy((dino_rep,dino_ids),(dino_label_c,dino_ids_c))
dino_tpcf = AstroMLmod.TPCF_score(dino_rep)
print(dino_tpcf)
dono_id = AstroMLmod.id_score(dino_rep)
print(dono_id)

knns.append(dino_knn)
tpcfs.append(dino_tpcf)
cas.append(dc_accuracy)
ids.append(dono_id)


##din huge


dinov3_vith = torch.hub.load("../dinov3", 'dinov3_vith16plus',
                                      source='local',
                                      weights="dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth")
print(dinov3_vith)
dinov3_vith_rep, dinov3_vith_label, dinov3_vith_ids = Custom.get_representations(model  = dinov3_vith,
                           loader = train,
                           encoder = True,
                           labeled = False)

dinov3_vith_rep_c, dinov3_vith_label_c, dinov3_vith_ids_c = Custom.get_representations(model  =dinov3_vith,
                           loader = class_,
                           encoder = True,
                           labeled = True)

#umap_  =v

#viz.umap(dinov3_vith_rep)

dinov3_vith_pca_var,dinov3_vith_pca_dim = viz.pca(dinov3_vith_rep,return_variance_dimension = True)
dinov3_vith_knn = test.KNN_accuracy(dinov3_vith_rep_c, dinov3_vith_label_c)
print(dinov3_vith_knn)
dinov3_vithc_accuracy = test.clustering_accuracy((dinov3_vith_rep,dinov3_vith_ids),(dinov3_vith_label_c,dinov3_vith_ids_c))
dinov3_vith_tpcf = AstroMLmod.TPCF_score(dinov3_vith_rep)
print(dinov3_vith_tpcf)
dinov3_vith_id = AstroMLmod.id_score(dinov3_vith_rep)
print(dinov3_vith_id)



knns.append(dinov3_vith_knn)
tpcfs.append(dinov3_vith_tpcf)
cas.append(dinov3_vithc_accuracy)
ids.append(dinov3_vith_id)


#ConvNext base

weights = tv.models.ConvNeXt_Base_Weights.IMAGENET1K_V1

imnet_convnext_base = tv.models.convnext_base(weights = weights)
imnet_convnext_base.classifier[-1] = torch.nn.Identity()

imnet_rep, imnet_label, imnet_ids = Custom.get_representations(model  = imnet_convnext_base,
                           loader = train,
                           encoder = True,
                           labeled = False)

imnet_rep_c, imnet_label_c, imnet_ids_c = Custom.get_representations(model  = imnet_convnext_base,
                           loader = class_,
                           encoder = True,
                           labeled = True)

#umap_  =viz.umap(imnet_rep)

imnet_pca_var,imnet_pca_dim = viz.pca(imnet_rep,return_variance_dimension = True)
imnet_knn = test.KNN_accuracy(imnet_rep_c, imnet_label_c)
print(imnet_knn)
ic_accuracy = test.clustering_accuracy((imnet_rep,imnet_ids),(imnet_label_c,imnet_ids_c))
imnet_tpcf = AstroMLmod.TPCF_score(imnet_rep)
print(imnet_tpcf)
imnet_id = AstroMLmod.id_score(imnet_rep)
print(imnet_id)

knns.append(imnet_knn)
tpcfs.append(imnet_tpcf)
cas.append(ic_accuracy)
ids.append(imnet_id)


#resnet

weights = tv.models.ResNet18_Weights.IMAGENET1K_V1 


imnet_resnet18 = tv.models.resnet18(weights = weights)

imnet_resnet18.fc = torch.nn.Identity()

res_imnet_rep, res_imnet_label, res_imnet_ids = Custom.get_representations(model  = imnet_resnet18,
                           loader = train,
                           encoder = True,
                           labeled = False)

res_imnet_rep_c, res_imnet_label_c, res_imnet_ids_c = Custom.get_representations(model  = imnet_resnet18,
                           loader = class_,
                           encoder = True,
                           labeled = True)

#umap_  =viz.umap(imnet_rep)

importlib.reload(AstroMLmod)
importlib.reload(test)
res_imnet_pca_var,res_imnet_pca_dim = viz.pca(res_imnet_rep,return_variance_dimension = True)
res_imnet_knn = test.KNN_accuracy(res_imnet_rep_c, res_imnet_label_c)
print("KNN: ",res_imnet_knn)
res_ic_accuracy = test.clustering_accuracy((res_imnet_rep,res_imnet_ids),(res_imnet_label_c,res_imnet_ids_c))
res_imnet_tpcf = AstroMLmod.TPCF_score(res_imnet_rep)
print("TPCF score: ",res_imnet_tpcf)
res_imnet_id = AstroMLmod.id_score(res_imnet_rep)
print("ID :",res_imnet_id)


knns.append(res_imnet_knn)
tpcfs.append(res_imnet_tpcf)
cas.append(res_ic_accuracy)
ids.append(res_imnet_id)




