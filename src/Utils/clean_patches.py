import os 
import cv2
import numpy as np 
from Utils.Processing import Processing

class PatchCleaner():
    """ This class is used to clean the patches of the BVD dataset.
    """
    def __init__(self) -> None:
        self.DIR = "patches/patches_bvd_clustd"
        self.CLUSTER_MAP = {
            1: "001B_clustd",
            2: "0024B_clustd",
            3: "003DEF_clustd"
        }
        self.processing = Processing()

    def clean_cluster(self, num_cluster):
        """Clean a cluster of patches.
        @param num_cluster: The number of the cluster.
        """
        patches = os.listdir(f"patches/patches_bvd_clustd/{self.CLUSTER_MAP[num_cluster]}")
        for patch in patches:
            patch_path = f"{self.DIR}/{self.CLUSTER_MAP[num_cluster]}/{patch}"
            image_patch = cv2.imread(patch_path)
            _, eosin, vessel_mask_cleaned = self.processing.process_patch(image_patch, use_hematoxylin=False, min_size=100)
            eosin = (eosin * 255).astype(np.uint8)
            cv2.imwrite(f"patches/patches_bvd_clustd_cleaned/{self.CLUSTER_MAP[num_cluster]}/{patch}", vessel_mask_cleaned)
            cv2.imwrite(f"patches/patches_bvd_clustd_HE/{self.CLUSTER_MAP[num_cluster]}/{patch}", eosin)

    def clean_patches(self):
        self.clean_cluster(1)
        self.clean_cluster(2)
        self.clean_cluster(3)
