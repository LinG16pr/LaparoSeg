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



class DresdenDataset(Dataset):

    def __init__(
        self,
        data_dir,
        history_length=4,
        augment=True,
        img_size=(80, 80)
    ):
        """
        Initializes the Dresden Surgical Anatomy Dataset (DSAD) from a given
        path to DSAD's multilabel dataset folder. 

        :param str data_dir: Root directory of the multilabel dataset folder. 
                             Both image and mask are in .png format.
        :param int history_length: The number of history frames included as 
                                   reference. Default 4 frames. 
        :param bool augment: Whether or not to strengthen the dataset by
                             performing random manipulations on the image. 
                             Default True. 
        :param tuple img_size: Image output size (H in pixels, W in pixels)
                               Default (80, 80)
        """
        self.data_dir = data_dir
        self.history_length = history_length
        self.augment = augment
        self.img_size = img_size
        self.TARGET_ORGANS = [
            "abdominal_wall",
            "colon",
            "liver",
            "pancreas",
            "small_intestine",
            "spleen",
            "stomach"
            # Background as the 8th channel
        ]
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

        self.video_frames = self._organize_files()
        self.sequences = self._generate_sequences()

    def _organize_files(self):
        """
        Organize input files (image and mask) based on the structure of DSAD. 

        DSAD Multilabel folder Structure:
        multilabel/                  # Passed to this class as data_dir
        ├── 02/                      # Surgery ID 02
        │   ├── image00.png          # Original Image
        │   ├── mask00_abdominal_wall.png  # Abdominal Wall Mask
        │   ├── mask00_colon.png           # Colon Mask
        │   ├── ...                  # Other organ masks (<=11 in total)
        │   ├── image01.png
        │   ├── mask01_abdominal_wall.png
        │   ├── ...
        │   └── weak_labels.csv      # Week label
        ├── 03/                      # Surgery ID 03
        │   ├── image00.png
        │   ├── mask00_abdominal_wall.png
        │   ├── ...
        └── ...                      # Other Surgery folders (02-31)

        :return: Dict with key as surgery_id, value as a list of frame info
                 Every frame_info is a dict with the following information:
                 {
                    'image_path': 'path/to/imageXX.png',
                    'mask_paths': ['path/to/maskXX_organ1.png', ...],
                    'missing_organs': ['colon', 'liver', ...],
                    'surgery_id': 'surgery_id'
                 }
        """

        video_dict = {}

        surgery_folders = glob.glob(os.path.join(self.data_dir, "*"))
        surgery_folders = [f for f in surgery_folders if os.path.isdir(f)]

        for surgery_folder in surgery_folders:
            surgery_id = int(os.path.basename(surgery_folder))

            # Load images as a list of strings, then sort by frame number
            image_files = glob.glob(os.path.join(surgery_folder, "image*.png"))
            image_files.sort(key=lambda x: int(os.path.basename(x)[5:-4]))

            # Collect masks for each image
            frame_dict = {}
            for image_file in image_files:
                frame_num = os.path.basename(image_file)[5:-4]

                # Collect masks by target channel order
                mask_files = []
                missing_organs = []
                for organ in self.TARGET_ORGANS:
                    organ_mask_file = os.path.join(surgery_folder, f"mask{frame_num}_{organ}.png")
                    if os.path.exists(organ_mask_file):
                        mask_files.append(organ_mask_file)
                    else:
                        missing_organs.append(organ)

                frame_dict[int(frame_num)] = {
                    "image_path": image_file,
                    "mask_paths": mask_files,
                    "missing_organs": missing_organs,
                    "surgery_id": surgery_id
                }

            if surgery_id not in video_dict:
                video_dict[surgery_id] = []

            sorted_frames = sorted(frame_dict.items(), key=lambda x: x[0])
            video_dict[surgery_id] = [item[1] for item in sorted_frames]

        return video_dict

    def _generate_sequences(self):
        """
        Generating training sequences, considering the continuity of frames. 
        """
        sequences = []

        for frame_info in self.video_frames.values():
            # Eliminate surgeries with frames less than history_length
            if len(frame_info) < self.history_length + 1:
                continue

            for i in range(self.history_length, len(frame_info)):
                # Combine every history_length frames as a sequence
                original_seq = frame_info[i-self.history_length:i+1]
                sequences.append(original_seq)
                if self.augment:
                    # Augmentations with duplication
                    if random.random() < 0.20:  # 20% chance for rotation
                        sequences.append((original_seq, "rotate"))
                    elif random.random() < 0.35:  # 15% chance for brightness/contrast
                        sequences.append((original_seq, "brightness_contrast"))
                    elif random.random() < 0.45:  # 10% chance for Gaussian noise
                        sequences.append((original_seq, "gaussian_noise"))

        return sequences

    def _apply_augmentation(
        self,
        frames,
        masks,
        augmentation_type
    ):
        """
        Adding noises to 
        """
        if augmentation_type == "rotate":
            angle = random.choice([90, 180, 270])
            frames = [TF.rotate(frame, angle) for frame in frames]
            masks = [TF.rotate(mask, angle) for mask in masks]
            masks = [self._recalculate_background(mask) for mask in masks]
        elif augmentation_type == "brightness_contrast":
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.85, 1.15)
            frames = [TF.adjust_brightness(frame, brightness_factor) for frame in frames]
            frames = [TF.adjust_contrast(frame, contrast_factor) for frame in frames]
        elif augmentation_type == "gaussian_noise":
            frames = [frame + torch.randn_like(frame) * 0.025 for frame in frames]
        return frames, masks

    def _recalculate_background(self, mask):
        """
        Making sure background still accurate after rotation.
        Risk comes from resizing.
        """
        mask = mask.clone()
        organs_sum = torch.sum(mask[:7], dim=0)
        mask[7] = 1.0 - torch.clamp(organs_sum, 0, 1)
        return mask

    def _load_masks(self, mask_paths, missing_organs):
        """
        Load a single frame's masks. Set to zero matrix for organs whose masks
        do not exist. Dynamically generates a background as the 8th channel.
        """
        h, w = self.img_size
        full_mask = np.zeros((8, h, w), dtype=np.float32)

        for i, organ in enumerate(self.TARGET_ORGANS):
            if organ not in missing_organs:
                mask_path = next(p for p in mask_paths if f"_{organ}.png" in p)
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask.resize((w, h), resample=Image.Resampling.NEAREST))
                mask = mask.astype(np.float32)
                full_mask[i] = mask

        organs_sum = np.sum(full_mask[:7], axis=0)
        full_mask[7] = 1.0 - np.clip(organs_sum, 0, 1)

        assert full_mask.shape[0] == 8, \
        f"Mask channel number not 8, actual number: {full_mask.shape[0]}"

        return torch.from_numpy(full_mask)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(
        self,
        idx
    ):
        """
        Get a sequence (frames with length history_length) by index. 
        Returns frames and masks
        """
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

            mask = self._load_masks(
                mask_paths=frame_info["mask_paths"],
                missing_organs=frame_info.get("missing_organs", [])
            )
            mask = TF.resize(mask, self.img_size)

            frames.append(img)
            masks.append(mask)

        if augmentation_type:
            frames, masks = self._apply_augmentation(
                frames,
                masks,
                augmentation_type
            )

        return frames, masks
