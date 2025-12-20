import backbone.Custom as Custom
import torch
from torch.utils.data import Dataset, DataLoader

def Galaxy_zoo_data_loaders(galaxyzoo_dir = "/idia/projects/camil/Koketso/galaxyzoo2",
                            galaxyzooq_dir = "/idia/projects/camil/Koketso/galaxyzoo/resized/galaxy_zoo_class_new",
                              valsplit = 0.2,
                            train_split = 0.8,
                            num_workers = 30,
                            batch_size = 128,
                            resize = 224,
                            crop_size = 224):

    dataset = Custom.dataset(galaxyzoo_dir)
    names = [name[0].split('/')[-1] for name in dataset.imgs]

    #classification validation

    classification_val_dataset = Custom.dataset(galaxyzooq_dir)

    datasets = Custom.train_val_dataset(dataset, 
                                        val_split = valsplit
                                        ,train_size = train_split)

    #Traning

    transformed_train_dataset = Custom.Custom(datasets['train'],
                                            names = names,
                                            resize = resize,
                                           crop = crop_size,
                                           )


    loader = DataLoader(transformed_train_dataset, 
                            batch_size, 
                            shuffle = True,
                            num_workers = num_workers)

    #validation

    transformed_val_dataset = Custom.Custom(datasets['val'],
                                            names = names,
                                            resize = resize,
                                           crop = crop_size,
                                           )

    val_loader = DataLoader(transformed_val_dataset, 
                            batch_size, 
                            shuffle = True,
                            num_workers = num_workers)


    #Classification validation

    transformed_classification_val_dataset = Custom.Custom_labelled(classification_val_dataset,
                                            names = names,
                                            resize = resize,
                                           crop = crop_size,
                                           )



    class_loader = DataLoader(transformed_classification_val_dataset, 
                            batch_size, 
                            shuffle = True,
                            num_workers = num_workers)

    return loader, val_loader, class_loader