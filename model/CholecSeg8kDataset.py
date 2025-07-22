import random
import glob
import os
import numpy as np
from PIL import Image

# PyTorch imports
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class CholecSeg8kDataset(Dataset):
    def __init__(
        self,
        data_dir,
        history_length=4,
        augment=True,
        img_size = (80, 80)
    ):
        """
        Initializes the CholecSeg8K Dataset from a given path to the dataset folder.

        :param str data_dir: Root directory of the dataset folder.
        :param int history_length: The number of history frames included as reference. Default 4 frames.
        :param bool augment: Whether or not to strengthen the dataset by performing random manipulations
                             on the image. Default True.
        :param tuple img_size: Image output size (H in pixels, W in pixels). Default (80, 80)
        """
        self.data_dir = data_dir
        self.history_length = history_length
        self.augment = augment
        self.img_size = img_size
        self.TARGET_ORGANS = [
            "abdominal_wall",
            "liver",
            "gastrointestinal_tract",
            "fat",
            "connective_tissue",
            "blood",
            "cystic_duct",
            "gallbladder",
            "hepatic_vein",
            "liver_ligament",
            # Background as the last channel
        ]
        self.CLASS_INFO = {
            0: {"name": "background", "pixel_value": 50}, 
            1: {"name": "abdominal_wall", "pixel_value": 11},
            2: {"name": "liver", "pixel_value": 21},
            3: {"name": "gastrointestinal_tract", "pixel_value": 13},
            4: {"name": "fat", "pixel_value": 12},
            5: {"name": "grasper", "pixel_value": 31},
            6: {"name": "connective_tissue", "pixel_value": 23},
            7: {"name": "blood", "pixel_value": 24},
            8: {"name": "cystic_duct", "pixel_value": 25},
            9: {"name": "L-hook_electrocautery", "pixel_value": 32},
            10: {"name": "gallbladder", "pixel_value": 22},
            11: {"name": "hepatic_vein", "pixel_value": 33},
            12: {"name": "liver_ligament", "pixel_value": 5}
        }

        self.ACTIVE_CLASSES = {
            class_id: info for class_id, info in self.CLASS_INFO.items()
        }
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

        self.video_frames = self._organize_files()
        self.sequences = self._generate_sequences()

    def _organize_files(self):
        """
        """
        video_dict = {}
        
        video_folders = glob.glob(os.path.join(self.data_dir, "video*"))
        video_folders = [f for f in video_folders if os.path.isdir(f)]

        for video_folder in video_folders:
            video_id = int(os.path.basename(video_folder).replace("video", ""))
            clip_folders = glob.glob(os.path.join(video_folder, "video*"))
            clip_folders = [f for f in clip_folders if os.path.isdir(f)]

            for clip_folder in clip_folders:
                clip_id = int(os.path.basename(clip_folder).split("_")[1])

                image_files = glob.glob(os.path.join(clip_folder, "frame*_endo.png"))
                image_files.sort(key=lambda x: int(x.split('_')[1].split("\\frame")[0]))

                frame_info_list = []
                for img_path in image_files:
                    base_name = os.path.basename(img_path).replace("_endo.png", "")
                    watershed_path = os.path.join(clip_folder, f"{base_name}_endo_watershed_mask.png")
                    frame_id = int(base_name.split('_')[1])

                    frame_info = {
                        "image_path": img_path,
                        "watershed_path": watershed_path,
                        "video_id": video_id,
                        "clip_id": clip_id,
                        "frame_id": frame_id
                    }
                    frame_info_list.append(frame_info)
                
                if frame_info_list:
                    video_dict[(video_id, clip_id)] = frame_info_list
        
        return video_dict

    def _generate_sequences(self):
        """
        """
        sequences = []
        
        for (video_id, clip_id), frame_info in self.video_frames.items():
            if len(frame_info) < self.history_length + 1:
                continue
                
            for i in range(self.history_length, len(frame_info)):
                original_seq = frame_info[i-self.history_length:i+1]
                sequences.append(original_seq)

                if self.augment:
                    if random.random() < 0.20:
                        sequences.append((original_seq, "rotate"))
                    elif random.random() < 0.35:
                        sequences.append((original_seq, "brightness_contrast"))
                    elif random.random() < 0.45:
                        sequences.append((original_seq, "gaussian_noise"))
        
        return sequences

    def _load_mask(self, watershed_path):
        h, w = self.img_size
        num_classes = len(self.CLASS_INFO)
        full_mask = np.zeros((num_classes, h, w), dtype=np.float32)

        mask_img = Image.open(watershed_path).convert("L")
        mask_array = np.array(mask_img.resize((w, h), resample=Image.NEAREST))
        
        unique_values = np.unique(mask_array)
        
        for class_id, info in self.CLASS_INFO.items():
            full_mask[class_id] = (mask_array == info["pixel_value"]).astype(np.float32)
        
        return torch.from_numpy(full_mask)
    
    def _apply_augmentation(self, frames, masks, augmentation_type):
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

    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        augmentation_type = None

        if isinstance(sequence_info, tuple):
            sequence, augmentation_type = sequence_info
        else:
            sequence = sequence_info
        
        frames = []
        masks = []

        for frame_info in sequence:
            img = Image.open(frame_info["image_path"]).convert("RGB")
            img = self.transform(img)

            mask = self._load_mask(frame_info["watershed_path"])
            mask = TF.resize(mask, self.img_size)

            frames.append(img)
            masks.append(mask)

        if augmentation_type:
            frames, masks = self._apply_augmentation(frames, masks, augmentation_type)
        
        return frames, masks