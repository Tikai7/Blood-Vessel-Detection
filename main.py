import os 
from src.clean_patches import PatchCleaner

def main():
    if not os.path.exists("patches/patches_bvd_clustd_cleaned"):
        os.makedirs("patches/patches_bvd_clustd_cleaned")
        patchCleaner = PatchCleaner()
        patchCleaner.clean_patches()



if __name__ == "__main__":
    main()