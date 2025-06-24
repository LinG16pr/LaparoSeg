import random
import glob
import os

# PyTorch imports
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset, random_split



class DresdenDataset(Dataset):

    def __init__(
        self,
        data_dir,
        history_length = 4,
        augment = True
    ):
        """
        Args:
            data_dir (str): répertoire contenant les fichiers '.pt'.
            history_length (int): nombre de frames passées (et leurs masques) à inclure par séquence.
            augment (bool): applique l'augmentation des données.
        """
        self.data_dir = data_dir
        self.history_length = history_length
        self.augment = augment

        # Charger et organiser les fichiers par vidéo
        self.video_frames = self._load_video_frames()

        # Générer toutes les séquences possibles
        self.sequences = self._generate_sequences()

    def _load_video_frames(self):
        files = glob.glob(os.path.join(self.data_dir, "*.pt"))
        video_dict = {}
        for file in files:
            filename = os.path.basename(file).replace('.pt', '')
            vid, frame = map(int, filename.split('_'))
            if vid not in video_dict:
                video_dict[vid] = []
            video_dict[vid].append((frame, file))
        for vid in video_dict:
            video_dict[vid].sort()
        return video_dict

    def _generate_sequences(self):
        sequences = []
        for vid, frames in self.video_frames.items():
            if len(frames) < self.history_length + 1:
                continue
            for i in range(self.history_length, len(frames)):
                original_seq = [frames[j][1] for j in range(i - self.history_length, i + 1)]
                sequences.append(original_seq)
                if self.augment:
                    # Augmentations avec duplication
                    if random.random() < 0.20:  # 20% de chance pour rotation
                        sequences.append((original_seq, "rotate"))
                    elif random.random() < 0.35:  # 15% de chance pour brightness/contrast
                        sequences.append((original_seq, "brightness_contrast"))
                    elif random.random() < 0.45:  # 10% de chance pour Gaussian noise
                        sequences.append((original_seq, "gaussian_noise"))
        return sequences

    def _apply_augmentation(
        self,
        frames,
        masks,
        augmentation_type
    ):
        
        if augmentation_type == "rotate":
            angle = random.choice([90, 180, 270])
            frames = [TF.rotate(frame, angle) for frame in frames]
            masks = [TF.rotate(mask, angle) for mask in masks]
        elif augmentation_type == "brightness_contrast":
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.85, 1.15)
            frames = [TF.adjust_brightness(frame, brightness_factor) for frame in frames]
            frames = [TF.adjust_contrast(frame, contrast_factor) for frame in frames]
        elif augmentation_type == "gaussian_noise":
            frames = [frame + torch.randn_like(frame) * 0.025 for frame in frames]
        return frames, masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(
        self,
        idx
    ):
        
        sequence_info = self.sequences[idx]
        augmentation_type = None
        if isinstance(sequence_info, tuple):
            sequence_files, augmentation_type = sequence_info
        else:
            sequence_files = sequence_info

        frames = []
        masks = []
        transform = transforms.Resize((80, 80))
        for file in sequence_files:
            data = torch.load(file)
            frames.append(transform(data['image']))
            mask = transform(data['mask'])
            # On ne conserve que certains canaux utiles
            mask = mask[[0, 1, 4, 5, 7, 8, 9, 11]]

            # TEST
            #zero_channel = torch.zeros(1, mask.shape[1], mask.shape[2])
            #mask = torch.cat((mask, zero_channel), dim = 0)

            masks.append(mask)

        if augmentation_type:
            frames, masks = self._apply_augmentation(frames, masks, augmentation_type)
        #print("------------------------")
        #print("Dresden:")
        #print(frames[0].shape)
        #print(masks[0].shape)
        return frames, masks