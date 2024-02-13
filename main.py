import cv2
from image_processing import Processing
import matplotlib.pyplot as plt


def main():
    processing = Processing()
    image_patch = cv2.imread("dataset/archive/img/0a5be90855e3.png")
    image_mask = cv2.imread("dataset/archive/mask/0a5be90855e3.png")
    # image_patch = Processing.to_patch(image_patch)
    plt.figure()
    plt.imshow(image_patch)
    plt.show()
    h,e,_,_ = processing.process_patch(image_patch, use_hematoxylin=True, min_size=100, get_vessel_count=None)
    print("Number of vessels in the image patch:", processing.num_vessels)
    processing.visualize_results()

    image_mask = image_patch*image_mask
    h_mask = h*image_mask[:,:,0]
    e_mask = e*image_mask[:,:,1]

    plt.figure()
    plt.subplot(141)
    plt.imshow(image_patch)
    plt.title('Original Image')
    plt.subplot(142)
    plt.imshow(h_mask)
    plt.title('Hematoxylin')
    plt.subplot(143)
    plt.imshow(e_mask)
    plt.title('Eosin')
    plt.subplot(144)
    plt.imshow(image_mask)
    plt.title('Original Image Masked with Vessel Mask')
    plt.show()



if __name__ == "__main__":
    main()