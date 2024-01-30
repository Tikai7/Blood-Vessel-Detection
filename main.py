import cv2
from image_processing import Processing

def main():
    processing = Processing()
    image_patch = cv2.imread("./images/4_1.jpg")
    processing.process_patch(image_patch, use_hematoxylin=True, min_size=100, get_vessel_count=True)
    print("Number of vessels in the image patch:", processing.num_vessels)
    processing.visualize_results()

if __name__ == "__main__":
    main()