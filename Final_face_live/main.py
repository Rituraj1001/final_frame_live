from side.V_image_frame_func import process_image
import cv2

if __name__ == "__main__":
    # Set default values for model_dir and device_id
    default_model_dir = "./resources/anti_spoof_models"
    default_device_id = 0

    # Path of the image you want to process
    image_path = "images/sample/image_F2.jpg"

    # Call the process_image function with default arguments
    processed_image, label, value = process_image(image_path, default_model_dir, default_device_id)
    
    # Save the processed image
    result_image_path = "result_image/result_image.jpg"
    cv2.imwrite(result_image_path, processed_image)
    print("Processed image saved as:", result_image_path)
