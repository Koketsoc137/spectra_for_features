import torch
import os
from pathlib import Path
from byol_pytorch import BYOL
import backbone.data_handle.Custom as Custom
import backbone.data_handle.GalaxyZoo as gz
import torchvision as tv
import kornia.augmentation as K
import kornia
import backbone.custom_metrics.AstroMLmod3 as AstroMLmod
import wandb
import pandas as pd
import yaml
from sklearn.neighbors import KNeighborsClassifier


def get_data_loaders(config):
    data_config = config["data"]
    training_config = config["training"]
    data_path = Path(__file__).resolve().parents[1] / data_config["dataset_path"]
    return gz.galaxyzoo10(
        train_split=data_config["train_split"],
        val_split=data_config["val_split"],
        batch_size=training_config["batch_size"],
        resize=data_config["resize"],
        crop=data_config["crop_size"],
        data_path=data_path,
        num_workers=training_config["num_workers"],
    )


def evaluate(learner, train_loader, val_loader, device, epoch, config):
    knn_config = config["knn"]
    path_config = config["paths"]
    output_dir = Path(__file__).resolve().parents[1] / path_config["output_dir"]
    artifact_prefix = path_config["artifact_prefix"]
    learner.eval()
    val_loss = 0.0
    val_features = []
    val_names = []
    val_labels = []
    train_features = []
    train_names = []
    train_labels = []

    with torch.no_grad():
        for images, labels, names in val_loader:
            images = images.to(device)
            val_loss += learner(images).item()
            embeddings = learner(images, return_embedding=True)[1]
            val_features.extend(embeddings.cpu().tolist())
            val_names.extend(names)
            val_labels.extend(labels.tolist())

        for images, labels, names in train_loader:
            embeddings = learner(images.to(device), return_embedding=True)[1]
            train_features.extend(embeddings.cpu().tolist())
            train_names.extend(names)
            train_labels.extend(labels.tolist())

    val_loss /= len(val_loader)
    knn = KNeighborsClassifier(n_neighbors=knn_config["neighbors"])
    knn.fit(train_features, train_labels)
    train_knn_score = knn.score(train_features, train_labels) * 100
    val_knn_score = knn.score(val_features, val_labels) * 100

    val_df = pd.DataFrame(val_features)
    val_df.insert(0, "Image_names", val_names)
    val_df.insert(1, "label", val_labels)
    val_df.to_csv(output_dir / f"{artifact_prefix}_val_representations_epoch_{epoch}.csv", index=False)
    train_df = pd.DataFrame(train_features)
    train_df.insert(0, "Image_names", train_names)
    train_df.insert(1, "label", train_labels)
    train_df.to_csv(output_dir / f"{artifact_prefix}_train_representations_epoch_{epoch}.csv", index=False)

    id_score, _ = AstroMLmod.id_score(val_features)
    tpcf_score, _ = AstroMLmod.TPCF_score(val_features)
    return val_loss, train_knn_score, val_knn_score, id_score, tpcf_score


def train_byol(config):
    training_config = config["training"]
    data_config = config["data"]
    model_config = config["model"]
    augmentation_config = config["augmentation"]
    path_config = config["paths"]
    output_dir = Path(__file__).resolve().parents[1] / path_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")
    model = tv.models.efficientnet_b0(
        weights=tv.models.EfficientNet_B0_Weights[model_config["weights"]]
    )
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, model_config["num_classes"])
    model.classifier[1].weight.data.normal_(0, 0.01)

    loader, val_loader = get_data_loaders(config)

    augment_fn = torch.nn.Sequential(
        Custom.RandomRotationWithCrop(
            degrees=augmentation_config["rotation_degrees"],
            crop_size=int(augmentation_config["rotation_crop_ratio"] * data_config["crop_size"]),
            output_dim=data_config["crop_size"],
            p=augmentation_config["rotation_probability"],
        ),
        kornia.augmentation.RandomVerticalFlip(p=augmentation_config["vertical_flip_probability"]),
        kornia.augmentation.RandomHorizontalFlip(p=augmentation_config["horizontal_flip_probability"]),
        kornia.augmentation.RandomResizedCrop(
            [data_config["crop_size"], data_config["crop_size"]],
            scale=augmentation_config["random_crop_scale"],
            p=augmentation_config["random_crop_probability"],
        ),
        K.RandomGaussianBlur(
            kernel_size=augmentation_config["gaussian_blur_kernel_size"],
            sigma=augmentation_config["gaussian_blur_sigma"],
            p=augmentation_config["gaussian_blur_probability"],
        ),
    )

    learner = BYOL(
        model,
        image_size=data_config["crop_size"],
        hidden_layer=model_config["representation_layer"],
        augment_fn=augment_fn,
    )
    device = torch.device(training_config["device"])
    learner = learner.to(device)

    opt = torch.optim.Adam(learner.parameters(), lr=training_config["learning_rate"])
    loss_history = []
    knn_history = []

    for epoch in range(training_config["epochs"]):

        loss_ = 0.0
        learner.train()
        for i,Images in enumerate(loader):
            Images = Images[0]
            #send imaged to device
            images = Images.to(device)
            #optain loss
            loss = learner(images)

            #optimization steps
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() #update moving average of target encoder
            loss_ += loss.item()
            loss_per_500 = loss_
            if i%5 ==0:
                print("Batch epoch :"+ str(epoch) + " Loss :" + str(loss.item()))

        train_loss = loss_ / len(loader)
        val_loss, train_knn_score, val_knn_score, id_score, tpcf_score = evaluate(
            learner, loader, val_loader, device, epoch, config
        )
        loss_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        pd.DataFrame(loss_history).to_csv(
            output_dir / f"{path_config['artifact_prefix']}_train_val_loss.csv", index=False
        )
        wandb.log({
            "Training epoch loss": train_loss,
            "Validation epoch loss": val_loss,
            "Train KNN score": train_knn_score,
            "Validation KNN score": val_knn_score,
            "ID score": float(id_score),
            "TPCF_score": float(tpcf_score),
        })
        print(f"Epoch [{epoch + 1}/{training_config['epochs']}], Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

        if val_loss < best_loss:

            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'augmentations':augment_fn,
                }, output_dir / path_config["best_checkpoint_name"])


        knn_history.append({
            "epoch": epoch,
            "train_knn_score": train_knn_score,
            "val_knn_score": val_knn_score,
        })
        pd.DataFrame(knn_history).to_csv(
            output_dir / f"{path_config['artifact_prefix']}_train_val_knn_scores.csv", index=False
        )

        torch.save({
                'knn_history':knn_history,

            'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'augmentations':augment_fn,
                'optimizer_state_dict': opt.state_dict(),
                }, output_dir / path_config["checkpoint_name"])


if __name__ == "__main__":
    config_path = Path(__file__).with_suffix(".yaml")
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)
    train_byol(config)
