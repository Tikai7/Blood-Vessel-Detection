import json
import numpy as np
import cv2
import os

class Masker:
    """ This class is used to generate the mask images from the json files.
    """
    
    def __init__(self) -> None:
        self.unmask_dir = "patches/patches_bvd_clustd"
        self.json_dir = "patches/patches_bvd_clustd_json"
        self.mask_dir = "patches/patches_bvd_clustd_mask"

        self.json_test_dir = "patches/patches_test/patches_anno_medecin_bvd_json"
        self.mask_test_dir = "patches/patches_test/patches_anno_medecin_bvd_mask"
        self.unmask_test_dir = "patches/patches_test/patches_anno_medecin_bvd"
        
    def gen_mask_img(self, json_filename, original_img_filename, mask_img_filename, write_dir=None):
        """ Generate a mask image from a json file and an original image.
        @param json_filename: The json filename.
        @param original_img_filename: The original image filename.
        @param mask_img_filename: The mask image filename.
        """
        # read json file
        with open(json_filename, "r") as f:
            data = f.read()
        # convert str to json objs
        data = json.loads(data)
        # read the original image
        image = cv2.imread(original_img_filename)
        if write_dir is not None:
            cv2.imwrite(write_dir, image)
        # create a mask
        mask = np.zeros_like(image, shape=(image.shape[0],image.shape[1]), dtype=np.uint8)
        # for all the shapes in the json file
        for shape in data["shapes"]:
            # get the points 
            points = shape["points"]
            points = np.array(points, dtype=np.int32)
            # fill the contour with 255
            cv2.fillPoly(mask, [points], (255, 255, 255))
        # save the mask
        cv2.imwrite(mask_img_filename, mask)

    def match_image_to_mask(self, cluster="001B_clustd"):
        """ Match the images to the mask for a given cluster.
        @param cluster: The cluster name.
        """
        mask_filenames = os.listdir(self.mask_dir + '/' + cluster)
        unmask_filenames = os.listdir(self.unmask_dir + '/' + cluster)  
        filtered_filenames = list(filter(lambda x: x in unmask_filenames, mask_filenames))
        for filename in filtered_filenames:
            image = cv2.imread(self.unmask_dir + '/' + cluster + '/' + filename)
            cv2.imwrite(f"C:/Cours Sorbonne/S2/PLDAC/Projet/dataset/img/{filename}",image)

    
    def build_mask(self):
        """ Build the mask images from the json files.
        """
        skip_clusters = ["001B_clustd", "0024B_clustd","003DEF_clustd"]
        json_dir = os.listdir(self.json_dir)
        for json_filename in json_dir:
            if json_filename in skip_clusters:
                continue
            print(f"Processing {json_filename}")
            # get the original image filename
            mask_clustered_dir = self.mask_dir + '/' + json_filename
            unmasked_clustered_dir = self.unmask_dir + '/' + json_filename
            json_clustered_dir = self.json_dir + '/' + json_filename
            json_filenames = os.listdir(json_clustered_dir)
            # iterate through the json files and generate the mask
            for json_filename in json_filenames:
                mask_img_filename = mask_clustered_dir + '/' + json_filename.replace('.json', '.png')
                original_img_filename = unmasked_clustered_dir + '/' + json_filename.replace('.json', '.png')
                json_filename = json_clustered_dir + '/' + json_filename
                self.gen_mask_img(json_filename, original_img_filename, mask_img_filename)


    def build_test_dataset(self):  
        json_dir = os.listdir(self.json_test_dir)
        for json_filename in json_dir:
            print(f"Processing {json_filename}")
            # get the original image filename
            cluster_of_json = json_filename.split("_")[2].split('.')[0]+'_clustd'
            mask_img_filename = self.mask_test_dir + '/' + json_filename.replace('.json', '.png')
            original_img_filename = self.unmask_dir + '/' + cluster_of_json + '/' + json_filename.replace('.json', '.png')
            original_img_filename_to_write = self.unmask_test_dir + '/' + json_filename.replace('.json', '.png')
            json_filename = self.json_test_dir + '/' + json_filename
            self.gen_mask_img(json_filename, original_img_filename, mask_img_filename, write_dir=original_img_filename_to_write)

                


masker = Masker()
masker.build_mask()
masker.match_image_to_mask(cluster="001B_clustd")
masker.match_image_to_mask(cluster="0024B_clustd")
masker.match_image_to_mask(cluster="003DEF_clustd")
masker.build_test_dataset()
