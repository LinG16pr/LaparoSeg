from scipy.ndimage import distance_transform_edt, binary_erosion
import matplotlib.pyplot as plt
import numpy as np
import warnings
import argparse
import random
import glob
import os

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset, random_split

from DMBMSNet import DMBMSNet
from DresdenDataset import DresdenDataset
from HemoSetDataset import HemoSetDataset
from CholecSeg8kDataset import CholecSeg8kDataset

torch.manual_seed(1)
cudnn.benchmark = True
warnings.simplefilter("ignore", category=FutureWarning)

def parse_args():

    parser = argparse.ArgumentParser(
        description = "Training PyTorch script for [name of the model]"
    )

    # Model arguments

    parser.add_argument("--num_classes",        type = int,  default = 8,     help = "Number of classes to segment.")
    parser.add_argument("--num_encoder_blocks", type = int,  default = 5,     help = "Number of encoder blocks in FrameEncoder (last one does not perform downsampling).")
    parser.add_argument("--num_heads",          type = int,  default = 8,     help = "Number of heads in attention mechanisms.")
    parser.add_argument("--k",                  type = int,  default = 5,     help = "")
    parser.add_argument("--memory_length",      type = int,  default = 4,     help = "Number of past frames to use as memory in the attention mechanisms.")
    parser.add_argument("--num_memory_blocks",  type = int,  default = 4,     help = "Number of attention blocks in MemoryAttention.")
    parser.add_argument("--deformable",         type = bool, default = True,  help = "Replace standard convolutions with deformable convolutions within FrameEncoder.")
    parser.add_argument("--use_crf",            type = bool, default = False, help = "Use a posterior CRF to smooth predictions.")

    # Training arguments

    parser.add_argument("--data_dir",           type = str,  required = True, help = "Path to the dataset directory.")
    parser.add_argument("--augmentation",       type = bool, default = False, help = "Augment training set.")
    parser.add_argument("--batch_size",         type = int,  default = 1,     help = "Batch size.")
    parser.add_argument("--epochs",             type = int,  default = 40,    help = "Maximum number of epochs (if no early stopping activated).")
    parser.add_argument("--lr",                 type = float,default = 0.0001,help = "Maximum learning rate.")
    parser.add_argument("--patience",           type = int,  default = 3,     help = "Number of epochs withoput improvement to wait before early stopping.")
    parser.add_argument("--device",             type = str,  default = "cuda",help = "Device to run the training on.")



# ------------------------- #
# Fonctions de calcul des métriques
# ------------------------- #

def accuracy_score(pred, gt):
    """ Calcule l'accuracy pixel par pixel. """
    return np.sum(pred == gt) / pred.size

def dice_score_per_sample(pred, gt, num_classes):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        denominator = np.sum(pred_c) + np.sum(gt_c)
        if denominator == 0:
            dice = 1.0
        else:
            dice = (2. * intersection + eps) / (denominator + eps)
        scores.append(dice)
    return np.mean(scores)

def jaccard_score_per_sample(pred, gt, num_classes):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        union = np.sum(pred_c) + np.sum(gt_c) - intersection
        if union == 0:
            score = 1.0
        else:
            score = (intersection + eps) / (union + eps)
        scores.append(score)
    return np.mean(scores)

def compute_hd95(binary1, binary2):
    """ Calcule la Hausdorff Distance 95 pour deux masques binaires. """
    struct = np.ones((3, 3), dtype=bool)
    edge1 = binary1 ^ binary_erosion(binary1, structure=struct)
    edge2 = binary2 ^ binary_erosion(binary2, structure=struct)
    if not edge1.any() or not edge2.any():
        return np.nan
    dt_edge2 = distance_transform_edt(~edge2)
    distances1 = dt_edge2[edge1]
    dt_edge1 = distance_transform_edt(~edge1)
    distances2 = dt_edge1[edge2]
    all_distances = np.concatenate([distances1, distances2])
    hd95 = np.percentile(all_distances, 95)
    return hd95

def hausdorff95_score_per_sample(pred, gt, num_classes):
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.bool_)
        gt_c = (gt == c).astype(np.bool_)
        if not pred_c.any() and not gt_c.any():
            hd95 = 0.0
        elif not pred_c.any() or not gt_c.any():
            hd95 = np.inf
        else:
            hd95 = compute_hd95(pred_c, gt_c)
        scores.append(hd95)
    # On moyenne uniquement sur les classes pour lesquelles la distance est finie
    finite_scores = [s for s in scores if np.isfinite(s)]
    if len(finite_scores) == 0:
        return np.inf
    return np.mean(finite_scores)




# ------------------------- #
# Boucle d'entraînement et évaluation
# ------------------------- #

def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_classes = 8,
    num_epochs = 20,
    lr = 1e-5,
    patience = 4,
    device = "cuda"
):
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        anneal_strategy='cos'
    )
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer = optimizer,
    #    factor = 0.1,
    #    patience = 2
    #)

    best_val_loss = np.inf
    patience_counter = 0
    best_model_state = None
    nan_detected = False

    train_loss_history = []
    val_loss_history = []

    epochs_counter = 0
    for epoch in range(num_epochs):
        epochs_counter += 1
        model.train()
        total_loss = 0
        images_counter = 0

        for images, masks in train_loader:
            images_counter += 1
            images = [img.squeeze(1).to(device) for img in images]
            masks  = [mask.squeeze(1).to(device) for mask in masks]
            optimizer.zero_grad()
            
            current_frame = images[-1]
            current_mask  = masks[-1]
            memory_frames = images[:-1]
            memory_masks  = masks[:-1]

            prediction = model(current_frame, training=True, memory_frames=memory_frames, memory_masks=memory_masks)
            loss = criterion(prediction, current_mask.argmax(dim=1).long())
            print(f"[{epochs_counter}/{num_epochs}] [{images_counter}/{len(train_loader)}] Training loss: {loss}")

            if torch.isnan(loss).any():
                print(f"⚠️ NaN loss detected during training at epoch n°{epoch+1}, batch n°{images_counter}.")
                nan_detected = True
                break

            loss.backward()
            optimizer.step()
            #scheduler.step()
            total_loss += loss.item()

            del current_frame, current_mask, memory_frames, memory_masks, prediction, loss
            torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            images_counter = 0
            for images, masks in val_loader:
                images_counter += 1
                images = [img.squeeze(1).to(device) for img in images]
                masks  = [mask.squeeze(1).to(device) for mask in masks]
                current_frame = images[-1]
                current_mask  = masks[-1]
                memory_frames = images[:-1]
                memory_masks  = masks[:-1]

                prediction = model(current_frame, training=True, memory_frames=memory_frames, memory_masks=memory_masks)
                loss = criterion(prediction, current_mask.argmax(dim=1).long())
                print(f"[{epochs_counter}/{num_epochs}] [{images_counter}/{len(val_loader)}] Validation loss: {loss}")

                if torch.isnan(loss).any():
                    print(f"⚠️ NaN loss detected during validation at epoch n°{epoch+1}, batch n°{images_counter}.")
                    nan_detected = True
                    break

                val_loss += loss.item()

                del current_frame, current_mask, memory_frames, memory_masks, prediction, loss
                torch.cuda.empty_cache()

        if nan_detected:
            break

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print("------------------------------")
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print("------------------------------")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss
            }, "best_model_checkpoint.pth")
            print("🔹 New best model found and saved !")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement since {patience_counter} epoch(s)...")

        if patience_counter >= patience:
            print(f"⏹️ Early stopping activated after {epoch+1} epochs. Best val loss = {best_val_loss:.4f}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("✅ Best model reloaded !")

        #plt.figure()
        #plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label = "Training Loss")
        #plt.plot(range(1, len(val_loss_history)+1),   val_loss_history,   label = "Validation Loss")
        #plt.xlabel("Epoch")
        #plt.ylabel("Loss")
        #plt.title("Évolution de la Training et Validation Loss")
        #plt.legend()
        #plt.show()

        model.eval()

        test_loss = 0
        accuracies = []
        dices = []
        jaccards = []
        hd95s = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = [img.squeeze(1).to(device) for img in images]
                masks  = [mask.squeeze(1).to(device) for mask in masks]
                current_frame = images[-1]
                current_mask  = masks[-1]
                memory_frames = images[:-1]
                memory_masks  = masks[:-1]

                prediction = model(current_frame, training=True, memory_frames=memory_frames, memory_masks=memory_masks)
                loss = criterion(prediction, current_mask.argmax(dim=1).long())
                test_loss += loss.item()

                # Récupération de la prédiction et du ground truth sous forme d'array (2D) pour chaque image du batch
                pred_masks = prediction.argmax(dim=1).detach().cpu().numpy()  # forme: (batch, H, W)
                target_masks = current_mask.argmax(dim=1).detach().cpu().numpy()  # forme: (batch, H, W)

                # Calcul des métriques pour chaque image du batch et moyenne
                batch_acc = []
                batch_dice = []
                batch_jaccard = []
                batch_hd95 = []
                for i in range(pred_masks.shape[0]):
                    acc = accuracy_score(pred_masks[i], target_masks[i])
                    dice = dice_score_per_sample(pred_masks[i], target_masks[i], num_classes)
                    jaccard = jaccard_score_per_sample(pred_masks[i], target_masks[i], num_classes)
                    hd95 = hausdorff95_score_per_sample(pred_masks[i], target_masks[i], num_classes)
                    batch_acc.append(acc)
                    batch_dice.append(dice)
                    batch_jaccard.append(jaccard)
                    batch_hd95.append(hd95)

                accuracies.append(sum(batch_acc) / len(batch_acc))
                dices.append(sum(batch_dice) / len(batch_dice))
                jaccards.append(sum(batch_jaccard) / len(batch_jaccard))
                hd95s.append(sum(batch_hd95) / len(batch_hd95))

                del current_frame, current_mask, memory_frames, memory_masks, prediction, loss
                torch.cuda.empty_cache()

        avg_test_loss = test_loss / len(test_loader)
        avg_accuracy = np.mean(accuracies)
        avg_dice = np.mean(dices)
        avg_jaccard = np.mean(jaccards)
        avg_hd95 = np.mean([v for v in hd95s if np.isfinite(v)])  # Moyenne uniquement sur les valeurs finies

        print("------------------------------")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Accuracy: {avg_accuracy:.4f}")
        print(f"Dice Coefficient: {avg_dice:.4f}")
        print(f"Jaccard Index: {avg_jaccard:.4f}")
        print(f"Hausdorff Distance 95: {avg_hd95:.4f}")
        print("------------------------------")

    print("🏁 Training finished !")



if __name__ == '__main__':

    args = parse_args()

    # Model arguments
    num_classes        = 8 #args.num_classes
    num_encoder_blocks = 5 #args.num_encoder_blocks
    num_heads          = 8 #args.num_heads
    k                  = 5 #args.k
    memory_length      = 4 #args.memory_length
    num_memory_blocks  = 4 #args.num_memory_blocks
    deformable         = True #args.deformable
    use_crf            = True #args.use_crf

    # Training arguments
    data_dir           = r"D:\archive\Dresden Dataset\dresden_tensors" #args.data_dir
    augmentation       = False #args.augmentation
    batch_size         = 2 #args.batch_size
    epochs             = 1000 #args.epochs
    lr                 = 1e-5 #args.lr
    patience           = 6 #args.patience
    device             = "cuda" #args.device

    # Dresden Dataset

    dataset_1 = DresdenDataset(
        data_dir,
        history_length = memory_length,
        augment = augmentation
    )
    
    train_vids_1 = {4, 5, 8, 10, 12, 16, 17, 22, 23, 24, 25, 27, 29, 30, 31}
    val_vids_1   = {2, 7, 11, 18, 20}
    test_vids_1  = {3, 21, 26}
    #train_vids_1 = {4, 5, 8, 10, 12, 16, 17, 22, 23, 24, 25, 27, 29, 30, 31, 3, 21, 26}
    #val_vids_1   = {2, 7, 11, 18, 20}
    #test_vids_1  = {}

    train_indices_1 = []
    val_indices_1   = []
    test_indices_1  = []

    for idx, seq in enumerate(dataset_1.sequences):
        # Chaque séquence est soit une liste de fichiers, soit un tuple (liste de fichiers, type d'augmentation)
        if isinstance(seq, tuple):
            files = seq[0]
        else:
            files = seq
        # On récupère l’ID vidéo depuis le nom du premier fichier de la séquence
        video_id = int(os.path.basename(files[0]).split('_')[0])
        if video_id in train_vids_1:
            train_indices_1.append(idx)
        elif video_id in val_vids_1:
            val_indices_1.append(idx)
        elif video_id in test_vids_1:
            test_indices_1.append(idx)

    train_set_1 = Subset(dataset_1, train_indices_1)
    val_set_1   = Subset(dataset_1, val_indices_1)
    test_set_1  = Subset(dataset_1, test_indices_1)

    # HemoSet Dataset

    dataset_2 = HemoSetDataset(
        data_dir = r"D:\HemoSet\hemoset_tensors",
        history_length = memory_length,
        augment = augmentation
    )

    train_vids_2 = {1, 2, 3, 4, 5, 6}
    val_vids_2 = {7, 9}
    test_vids_2 = {10, 11}
    #train_vids_2 = {1, 2, 3, 4, 5}
    #val_vids_2 = {10, 11}
    #test_vids_2 = {}

    train_indices_2 = []
    val_indices_2   = []
    test_indices_2  = []

    for idx, seq in enumerate(dataset_2.sequences):
        # Chaque séquence est soit une liste de fichiers, soit un tuple (liste de fichiers, type d'augmentation)
        if isinstance(seq, tuple):
            files = seq[0]
        else:
            files = seq
        # On récupère l’ID vidéo depuis le nom du premier fichier de la séquence
        video_id = int(os.path.basename(files[0]).split('_')[0])
        if video_id in train_vids_2:
            train_indices_2.append(idx)
        elif video_id in val_vids_2:
            val_indices_2.append(idx)
        elif video_id in test_vids_2:
            test_indices_2.append(idx)

    train_set_2 = Subset(dataset_2, train_indices_2)
    val_set_2   = Subset(dataset_2, val_indices_2)
    test_set_2  = Subset(dataset_2, test_indices_2)

    # CholecSeg8k Dataset

    dataset_3 = CholecSeg8kDataset(
        data_dir = r"D:\CholecSeg8k\cholecseg8k_tensors",
        history_length = memory_length,
        augment = augmentation
    )

    train_vids_3 = {1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28}
    val_vids_3 = {35, 37, 43}
    test_vids_3 = {48, 52, 55}
    #train_vids_3 = {}
    #val_vids_3 = {}
    #test_vids_3 = {1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28, 35, 37, 43, 48, 52, 55}

    train_indices_3 = []
    val_indices_3   = []
    test_indices_3  = []

    for idx, seq in enumerate(dataset_3.sequences):
        # Chaque séquence est soit une liste de fichiers, soit un tuple (liste de fichiers, type d'augmentation)
        if isinstance(seq, tuple):
            files = seq[0]
        else:
            files = seq
        # On récupère l’ID vidéo depuis le nom du premier fichier de la séquence
        video_id = int(os.path.basename(files[0]).split('_')[0])
        if video_id in train_vids_3:
            train_indices_3.append(idx)
        elif video_id in val_vids_3:
            val_indices_3.append(idx)
        elif video_id in test_vids_3:
            test_indices_3.append(idx)

    train_set_3 = Subset(dataset_3, train_indices_3)
    val_set_3   = Subset(dataset_3, val_indices_3)
    test_set_3  = Subset(dataset_3, test_indices_3)

    #

    train_set = ConcatDataset([train_set_1])
    val_set = ConcatDataset([val_set_1])
    test_set = ConcatDataset([test_set_1])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=True, num_workers=4)

    model = DMBMSNet(
        input_dim          = 3,
        num_classes        = num_classes,
        num_encoder_blocks = num_encoder_blocks,
        base_dim           = 64,
        num_heads          = num_heads,
        k                  = k,
        memory_frames      = memory_length,
        num_memory_blocks  = num_memory_blocks,
        deformable         = deformable,
        use_crf            = use_crf
    )

    train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        num_classes  = num_classes,
        num_epochs   = epochs,
        lr           = lr,
        patience     = patience,
        device       = device
    )