import numpy as np
import matplotlib.pyplot as plt
from skimage import color, exposure,morphology
class Processing():
    """
        This class is used to process the image patches of H&E images.
        - The image patch is first separated into hematoxylin and eosin components.
        - Then, the image is enhanced using histogram equalization.
        - The enhanced image is then thresholded to get the vessel mask.
        - The vessel mask is cleaned to remove noise.
        - The number of vessels in the image patch is then counted optionally.
        *Note: The vessel mask can be obtained using either the hematoxylin or eosin component.
    """

    def __init__(self) -> None:
        self.dim = None
        self.patch = None
        self.hematoxylin = None
        self.eosin = None
        self.h_enhanced = None
        self.e_enhanced = None
        self.vessel_mask = None
        self.vessel_mask_cleaned = None
        self.label_image = None
        self.threshold_value = None
        
    
    def process_patch(self, image_patch, use_hematoxylin=True, min_size=100):
        """
            This function is used to process the image patch.
            :params: image patch, use_hematoxylin, min_size, get_vessel_count
            - get_vessel_count: True if the number of vessels in the image patch is to be returned, False otherwise. (Useful for classification)
            - use_hematoxylin: True if hematoxylin component is to be used for thresholding, False if eosin component is to be used.
            - min_size: minimum size of the object to be retained.
            :return: enhanced hematoxylin and eosin components of the image patch, and optionally the number of vessels in the image patch.
        """
        self.patch = image_patch
        self.dim = image_patch.shape
        hematoxylin, eosin = self._color_decovolution(image_patch)
        h_enhanced, e_enhanced = self._image_enhancement(hematoxylin, eosin)
        vessel_mask = self._get_vessel_mask(h_enhanced) if use_hematoxylin else self._get_vessel_mask(e_enhanced)
        vessel_mask_cleaned = self._get_vessel_mask_cleaned(vessel_mask, min_size=min_size) 

        return h_enhanced, e_enhanced, vessel_mask_cleaned
    
    def _color_decovolution(self, image_patch):
        """
            This function is used to separate the image patch into hematoxylin and eosin components.
            :params: image patch.
            :return: hematoxylin and eosin components of the image patch.
        """
        ihc_hed = color.rgb2hed(image_patch)

        self.hematoxylin = ihc_hed[:, :, 0]
        self.eosin = ihc_hed[:, :, 1]

        return self.hematoxylin, self.eosin
        
    def _image_enhancement(self, hematoxylin, eosin):
        """
            This function is used to enhance the image using histogram equalization.
            :params: hematoxylin and eosin components of the image patch.
            :return: enhanced hematoxylin and eosin components.
        """
        self.h_enhanced = exposure.equalize_hist(hematoxylin)
        self.e_enhanced = exposure.equalize_hist(eosin)
        return self.h_enhanced, self.e_enhanced
    
    def _get_vessel_mask(self, blood_channel):
        """
            This function is used to threshold the image to get the vessel mask.
            :params: enhanced hematoxylin component of the image patch.
            :return: vessel mask.
        """
        self.threshold_value = np.min(blood_channel)
        self.vessel_mask = blood_channel > self.threshold_value
        return self.vessel_mask
    
    def _get_vessel_mask_cleaned(self, vessel_mask, min_size=100):
        """
            This function is used to clean the vessel mask.
            :params: vessel mask.
            :return: cleaned vessel mask.
        """
        self.vessel_mask_cleaned = morphology.remove_small_objects(vessel_mask, min_size=min_size)
        self.vessel_mask_cleaned = morphology.binary_closing(self.vessel_mask_cleaned)
        self.vessel_mask_cleaned = np.invert(self.vessel_mask_cleaned.astype(int)) + 2
        self.vessel_mask_cleaned = np.stack((self.vessel_mask_cleaned,)*self.dim[-1], axis=-1)

        return self.vessel_mask_cleaned
    
    def visualize_results(self):
        """This function is used to visualize the results.
        """
        image_masked = self.patch * self.vessel_mask_cleaned

        plt.figure(figsize=(12, 6))

        plt.subplot(231)
        plt.imshow(self.hematoxylin, cmap='gray')
        plt.title('Hematoxylin')

        plt.subplot(232)
        plt.imshow(self.eosin, cmap='gray')
        plt.title('Eosin')

        plt.subplot(233)
        plt.imshow(self.h_enhanced, cmap='gray')
        plt.title('Enhanced Hematoxylin')

        plt.subplot(234)
        plt.imshow(self.e_enhanced, cmap='gray')
        plt.title('Enhanced Eosin')

        plt.subplot(235)
        plt.imshow(image_masked, cmap='gray')
        plt.title('Vessel Mask (Hematoxylin)')

        plt.show()