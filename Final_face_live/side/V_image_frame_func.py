import cv2
import numpy as np
import argparse
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import os

def process_image(image_path, model_dir, device_id, display=True):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_path)  # Read the image from the specified path

    # Process the image
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    # Determine the label with the highest prediction score
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        result_text = "Real Face Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        result_text = "Fake Face Score: {:.2f}".format(value)
        color = (0, 0, 255)

    # Draw bounding box and result text on the processed image
    processed_image = image.copy()
    cv2.rectangle(processed_image, (image_bbox[0], image_bbox[1]),
                  (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]), color, 2)
    cv2.putText(processed_image, result_text, (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)

    if display:
        # Display the processed image
        cv2.imshow('Processed Image', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print the score and label
        print("Score:", value)
        if label == 1:
            print("Label: Real Face")
        else:
            print("Label: Fake Face")

    return processed_image, label, value  # Returning processed image, label, and value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for face liveness")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--device_id", type=int, default=0, help="GPU ID (0, 1, 2, ...)")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models",
                        help="Directory containing anti-spoofing models")
    args = parser.parse_args()

    # Process the input image and get processed image, label, and value
    processed_image, label, value = process_image(args.image_path, args.model_dir, args.device_id)
