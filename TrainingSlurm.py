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
import backbone.AstroMLmod3 as AstroMLmod
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
    # labels = perturb_list_by_swapping(labels, percentage=30)
    images = images.astype(np.float16)
    

    trainsformed_dataset = cust.ArrayDataset(images = images,
                                            labels =labels,
                                            names = ids,
                                            resize = 256,
                                            crop = 224,
                                            eval_mode =False )


    dataset_split = cust.train_val_dataset(transformed_dataset, train_size = 0.6,val_split=0.4)


    dataset_split['val'].dataset.eval_mode = True
    
    train_loader = torch.utils.data.DataLoader(dataset_split['train'], batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset_split['val'], batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
    
def evaluate(model, train_loader,test_loader,criterion,device):
            
    correct, total = 0, 0
    train_loss = 0
    with torch.no_grad():
        for images, labels,_ in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            #Computing loss
            train_loss += criterion(outputs,labels)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    print(f'Train Accuracy: {100 * correct / total:.2f}%')
    train_accuracy = 100 * correct / total

    correct, total = 0, 0
    val_loss = 0
    with torch.no_grad():
        for images, labels,_ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #Compute the loss
            val_loss += criterion(outputs,labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    test_accuracy = 100 * correct / total
    torch.cuda.empty_cache()
    #evaluate(model, train_loader, test_loader, device)


    return train_accuracy, test_accuracy, train_loss, val_loss



def train_resnet(num_epochs=100, learning_rate=0.0005, Dir ="galaxy_zoo_class_new", batch_size=64, device='cuda'):

    fig = plt.figure(dpi = 300)
    plt.style.use("default")
    plt.figure(figsize=(15,10))
    plt.rcParams.update({'font.size': 20}) 


    train_loader, test_loader = galaxyzoo10(batch_size = batch_size)
    
    model = models.efficientnet_b0(weights = "IMAGENET1K_V1")

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max = 100, 
                                                           eta_min=0)
    #get_representations(model = model,loader = train_loader, batch_size = batch_size, epoch = 0,device  = device)
    model.classifier[1]= nn.Linear(model.classifier[1].in_features, 10) 
    #randomize the weights of the newly added sub-layers
    model.classifier[1].weight.data.normal_(0,0.01)
    model = model.to(device)

    ID_scores = []
    train_val_accuracy_loss = []
    TPCF_scores = []

    epoch = 0


    #Obtain train and test accuracies
    
    train, val,train_loss,val_loss = evaluate(model, train_loader, test_loader,criterion, device)

    train_val_accuracy_loss.append((train, val,train_loss,val_loss))


    #get representatations
    test_representations, test_labels = cust.get_representations(model = model,loader = test_loader, batch_size = batch_size, epoch = 0,device  = device)
    train_representations,train_labels = cust.get_representations(model = model,loader = train_loader, batch_size = batch_size, epoch = 0,device  = device)


    #conpute the id_score
    id_score_test,std_test = AstroMLmod.id_score(test_representations)
    id_score_train,std_train = AstroMLmod.id_score(train_representations)
    
    ID_scores.append((id_score_test,std_test,id_score_train,std_train))

    #The two pont correlatiion function scores on training and test data

    TPCF_score_val = AstroMLmod.TPCF_score(test_representations, epoch = epoch)
    TPCF_score_train = AstroMLmod.TPCF_score(train_representations, epoch = epoch)


    TPCF_scores.append((TPCF_score_val,TPCF_score_train))

    
    #intrinsic_dimension.append((id_score,std))

    
    #Faltten the manifold

    pkl_filename = "plots/Test_representations_labels"+str(epoch)+".csv"
    with open(pkl_filename, 'wb') as file:
        pickle.dump((test_representations,test_labels),file)
            
    pkl_filename = "plots/Train_representations_labels"+str(epoch)+".csv"
    with open(pkl_filename, 'wb') as file:
        pickle.dump((train_representations,train_labels),file)

    
        
    fig = plt.figure(dpi = 300)
    plt.style.use("default")
    plt.figure(figsize=(15,10))
    print("Epoch: 0")


    
    for epoch in range(1,num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels, names in train_loader:
            
            
            images, labels = images.to(device), labels.to(device)
            """
            Training
            """
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
                
            """
            id_stuff here
            
            test_representations,test_labels = cust.get_representations(model = model,loader = test_loader, batch_size = batch_size, epoch = 0,device  = device)
            #id_score,std = AstroMLmod.id_score(test_representations)
            loss = loss*(id_score/100)
            """

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        model.eval()
        """
        Model evaluation
        """
        
        train, val, train_loss, val_loss = evaluate(model, train_loader, test_loader, criterion, device)
        train_val_accuracy_loss.append((train, val,train_loss,val_loss,running_loss))

        #save classification layer for next epoch
    
        class_layer = model.classifier
        
        test_representations,test_labels = cust.get_representations(model = model,loader = test_loader, batch_size = batch_size, epoch = 0,device  = device)
        train_representations,train_labels = cust.get_representations(model = model,loader = train_loader, batch_size = batch_size, epoch = 0,device  = device)


        #conpute the id_score
        id_score_test,std_test = AstroMLmod.id_score(test_representations)
        id_score_train,std_train = AstroMLmod.id_score(train_representations)

        ID_scores.append((id_score_test,std_test,id_score_train,std_train))

        #Two point correlation function scores
        TPCF_score_val = AstroMLmod.TPCF_score(test_representations, epoch = epoch)
        TPCF_score_train = AstroMLmod.TPCF_score(train_representations, epoch = epoch)


        TPCF_scores.append((TPCF_score_val,TPCF_score_train))



        #model.fc = nn.Linear(512, 10) 
        model.classifier = class_layer
        #model.fc = model.fc.to(device)
        x = np.arange(epoch+1)
        """
        plt.plot([a for a,b in norm_scores], label = "Chi score", color = "blue")
        plt.savefig("NormScores.png")           
        #plt.errorbar(x, y, yerr=yerr, fmt='o', color = "blue", capsize=1)


        plt.plot([100-a for a in validation_accuracy], label = "validation error")
        plt.plot([100-a for a in train_accuracy], label = "Train error")
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        plt.savefig("Training_val.png")   
        """

        if epoch%10 ==0:
            pkl_filename = "Train_test_TPCF_score.csv"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(TPCF_scores,file)
                
            pkl_filename = "Train_val_accuracy_loss.csv"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(train_val_accuracy_loss,file)
                
            pkl_filename = "Train_val_id_score.csv"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(ID_scores,file) 

        pkl_filename = "plots/Test_representations_labels"+str(epoch)+".csv"
        with open(pkl_filename, 'wb') as file:
            pickle.dump((test_representations,test_labels),file)
            
        pkl_filename = "plots/Train_representations_labels"+str(epoch)+".csv"
        with open(pkl_filename, 'wb') as file:
            pickle.dump((train_representations,train_labels),file)


if __name__ == "__main__":
    train_resnet()
