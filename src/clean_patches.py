import cv2
from src.image_processing import Processing
import os 
import matplotlib.pyplot as plt

class PatchCleaner():
    def __init__(self) -> None:
        self.DIR = "patches/patches_bvd_clustd"
        self.CLUSTER_MAP = {
            1: "001B_clustd",
            2: "0024B_clustd",
            3: "003DEF_clustd"
        }
        self.processing = Processing()

    def clean_cluster(self, num_cluster):
        patches = os.listdir(f"patches/patches_bvd_clustd/{self.CLUSTER_MAP[num_cluster]}")
        for patch in patches:
            patch_path = f"{self.DIR}/{self.CLUSTER_MAP[num_cluster]}/{patch}"
            image_patch = cv2.imread(patch_path)
            _, _, vessel_mask_cleaned = self.processing.process_patch(image_patch, use_hematoxylin=False, min_size=100)
            cv2.imwrite(f"patches/patches_bvd_clustd_cleaned/{self.CLUSTER_MAP[num_cluster]}/{patch}", vessel_mask_cleaned)
            

    def clean_patches(self):
        self.clean_cluster(1)
        self.clean_cluster(2)
        self.clean_cluster(3)