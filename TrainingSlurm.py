import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import backbone.Custom as cust
import backbone.VISUAL as viz
import importlib
import matplotlib.pyplot as plt
import backbone.AstroMLmod as AstroMLmod
import numpy as np
import backbone.TwoNN as TwoNN
import time
import h5py
import skdim
import pickle
import random    


def get_data_loaders(Dir = "some/directory", batch_size = 32):

    #obtain  data from a folder a of images
    dataset = torchvision.datasets.ImageFolder(Dir)
    names = [name[0].split('/')[-1] for name in dataset.imgs]
    transformed_dataset = cust.Custom_labelled(dataset,names =names,resize = 256,crop = 224)


    dataset_split = cust.train_val_dataset(transformed_dataset, val_split=0.005)
    

    train_loader = torch.utils.data.DataLoader(dataset_split['train'], batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset_split['val'], batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

def perturb_list_by_swapping(lst, percentage=5):
    num_swaps = max(1, int(len(lst) * (percentage / 100)))  # Ensure at least one swap
    perturbed_lst = lst[:]  # Copy the list to avoid modifying the original

    for _ in range(num_swaps):
        i, j = random.sample(range(len(lst)), 2)  # Pick two distinct indices
        perturbed_lst[i], perturbed_lst[j] = perturbed_lst[j], perturbed_lst[i]  # Swap them
    
    return perturbed_lst


def galaxyzoo10(batch_size = 256):

    # To get the images and labels from file
    with h5py.File('Galaxy10_DECals.h5', 'r') as F:
        images = np.array(F['images'])
        labels = np.array(F['ans'])
        ids = np.array(F['ra'])
    
    # To convert the labels to categorical 10 classes

    # To convert to desirable type
    labels = labels.astype(np.int64)
    #labels = perturb_list_by_swapping(labels, percentage=30)
    images = images.astype(np.float16)
    

    transformed_dataset = cust.ArrayDataset(images = images,labels =labels,names = ids,resize = 256,crop = 224)


    dataset_split = cust.train_val_dataset(transformed_dataset, train_size = 0.6,val_split=0.4)
    

    train_loader = torch.utils.data.DataLoader(dataset_split['train'], batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset_split['val'], batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
    
def evaluate(model, train_loader,test_loader, device):
            
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels,_ in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    print(f'Train Accuracy: {100 * correct / total:.2f}%')
    train_accuracy = 100 * correct / total

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels,_ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    test_accuracy = 100 * correct / total
    torch.cuda.empty_cache()
    #evaluate(model, train_loader, test_loader, device)


    return train_accuracy, test_accuracy




def train_resnet(num_epochs=100, learning_rate=0.0005, Dir ="galaxy_zoo_class_new", batch_size=128, device='cuda'):

    fig = plt.figure(dpi = 300)
    plt.style.use("default")
    plt.figure(figsize=(15,10))
    plt.rcParams.update({'font.size': 20}) 


    train_loader, test_loader = galaxyzoo10(batch_size = batch_size)
    
    model = models.efficientnet_b0(weights = "IMAGENET1K_V1")

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #get_representations(model = model,loader = train_loader, batch_size = batch_size, epoch = 0,device  = device)
    model.classifier[1]= nn.Linear(model.classifier[1].in_features, 10) 
    #randomize the weights of the newly added sub-layers
    model.classifier[1].weight.data.normal_(0,0.01)
    model = model.to(device)

    intrinsic_dimension = []
    validation_accuracy = []
    train_accuracy = []
    norm_scores = []
    chi_scores = []

    #Obtain train and test accuracies
    
    train, val = evaluate(model, train_loader, test_loader, device)

    #get representatations
    test_representations, test_labels = cust.get_representations(model = model,loader = test_loader, batch_size = batch_size, epoch = 0,device  = device)

    #conpute the id_score
    #id_score,std = AstroMLmod.id_score(test_representations)
    
    #intrinsic_dimension.append((id_score,std))
    
    validation_accuracy.append(val)
    train_accuracy.append(train)
    #Faltten the manifold
    epoch = 0
    #val_flat = viz.umap(test_representations,scatter = True,name = "UMAP", dim = 2, min_dist = 0.0, n_neighbors = 15,alpha = 0.2)
    val_flat = viz.pca(test_representations,n_components = 2)

    print(len(val_flat))
    
    chi_score, norm_score = AstroMLmod.correlate_and_plot(val_flat,
                                                                      min_dist = 0.0,
                                                                      max_dist =1.5,
                                                                      label = "Correlation on flat manifold for epoch:"+str(epoch),
                                                                    fig_name = "plots/2PCR@Epoch: "+str(epoch),
                                                                  representations = test_representations
                                                             )
    #Append to list for later

    norm_scores.append(norm_score)
    chi_scores.append(chi_score)

        
    fig = plt.figure(dpi = 300)
    plt.style.use("default")
    plt.figure(figsize=(15,10))


    
    for epoch in range(1,num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels, names in train_loader:
            
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        model.eval()
        
        train, val = evaluate(model, train_loader, test_loader, device)
        #save classification layer for next epoch
    
        class_layer = model.classifier
        
        test_representations,test_labels = cust.get_representations(model = model,loader = test_loader, batch_size = batch_size, epoch = 0,device  = device)

        #conpute the id_score
        #id_score,std = AstroMLmod.id_score(test_representations)
    
        #intrinsic_dimension.append((id_score,std))
        #y, yerr = zip(*intrinsic_dimension)


        #Faltten the manifold
        val_umap = viz.umap(test_representations,scatter = True,name = "UMAP", dim = 2, min_dist = 0.0, n_neighbors = 15,alpha = 0.2)



        pkl_filename = "plots/val_flat"+str(epoch)+".csv"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(val_umap,file)

            
        val_flat = viz.pca(test_representations,n_components = 2)


        chi_score, norm_score = AstroMLmod.correlate_and_plot(val_flat,
                                                                      min_dist = 0.0,
                                                                      max_dist =1.5,
                                                                      label = "Correlation on flat manifold for epoch:"+str(epoch),
                                                                      fig_name = "plots/2PCR@Epoch: "+str(epoch),
                                                                     representations = test_representations)


        norm_scores.append(norm_score)
        chi_scores.append(chi_score)



        validation_accuracy.append(val)
        train_accuracy.append(train)

        #model.fc = nn.Linear(512, 10) 
        model.classifier = class_layer
        #model.fc = model.fc.to(device)
        x = np.arange(epoch+1)
        plt.plot([a for a,b in norm_scores], label = "Chi score", color = "blue")
        plt.savefig("NormScores.png")           
        #plt.errorbar(x, y, yerr=yerr, fmt='o', color = "blue", capsize=1)


        plt.plot([100-a for a in validation_accuracy], label = "validation error")
        plt.plot([100-a for a in train_accuracy], label = "Train error")
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        plt.savefig("Training_val.png")       

        if epoch%10 ==0:
            pkl_filename = "chi_scores.csv"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(chi_scores,file)
                
            pkl_filename = "norm_scores.csv"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(norm_scores,file)
                
            pkl_filename = "chi_validation.csv"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(validation_accuracy,file)
                
            pkl_filename = "chi_train.csv"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(train_accuracy,file) 


if __name__ == "__main__":
    train_resnet()
