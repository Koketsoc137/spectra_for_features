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

def galaxyzoo10(batch_size = 256
               train_size = 0.7,
               val_split = 0.3,
                resize = 224,
                crop = 224,
                ):

    # To get the images and labels from file
    with h5py.File('Galaxy10_DECals.h5', 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])
        ids = np.array(F['ra'])
    
    # To convert the labels to categorical 10 classes

    # To convert to desirable type
    labels = labels.astype(np.int64)
    #labels = perturb_list_by_swapping(labels, percentage=5)
    images = images.astype(np.float16)
    

    transformed_dataset = cust.ArrayDataset(images = images,labels =labels,names = ids,resize = 224,crop = 224)


    dataset_split = cust.train_val_dataset(transformed_dataset, train_size = 0.7,val_split=0.3)
    

    train_loader = torch.utils.data.DataLoader(dataset_split['train'], batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset_split['val'], batch_size=batch_size, shuffle=True)

    return train_loader, test_loader