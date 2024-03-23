import json
import numpy as np
import cv2
import os

class Masker:
    def __init__(self) -> None:
        self.unmask_dir = "patches/patches_bvd_clustd"
        self.json_dir = "patches/patches_bvd_clustd_json"
        self.mask_dir = "patches/patches_bvd_clustd_mask"
        
    def gen_mask_img(self, json_filename, original_img_filename, mask_img_filename):
        # read json file
        with open(json_filename, "r") as f:
            data = f.read()
        # convert str to json objs
        data = json.loads(data)
        # read the original image
        image = cv2.imread(original_img_filename)
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
        mask_filenames = os.listdir(self.mask_dir + '/' + cluster)
        unmask_filenames = os.listdir(self.unmask_dir + '/' + cluster)  
        filtered_filenames = list(filter(lambda x: x in unmask_filenames, mask_filenames))
        for filename in filtered_filenames:
            image = cv2.imread(self.unmask_dir + '/' + cluster + '/' + filename)
            cv2.imwrite(f"C:/Cours Sorbonne/S2/PLDAC/Projet/dataset/archive/img/{filename}",image)

    
    def build_mask(self):
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

masker = Masker()
# masker.build_mask()
# masker.match_image_to_mask(cluster="001B_clustd")
# masker.match_image_to_mask(cluster="0024B_clustd")
masker.match_image_to_mask(cluster="003DEF_clustd")
