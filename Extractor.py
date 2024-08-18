import cv2
import numpy as np
import pytesseract
import re
from matplotlib import pyplot as plt

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image at path {image_path} could not be loaded.")
    return image


def find_graph_contour(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No green contours found in the image.")
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)


def extract_y_axis_labels(image, x, y, h, padding_left=50, scale_factor=2):
    if x - padding_left < 0:
        left_crop = image[y:y + h, 0:x]
        left_x = 0
    else:
        left_crop = image[y:y + h, x - padding_left:x]
        left_x = x - padding_left

    scaled_left_crop = cv2.resize(left_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray_scaled_left_crop = cv2.cvtColor(scaled_left_crop, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.adaptiveThreshold(gray_scaled_left_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
    scaled_image = cv2.resize(thresholded_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    enhanced_image = cv2.convertScaleAbs(scaled_image, alpha=1.5, beta=0)
    sharpening_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(enhanced_image, -1, sharpening_filter)

    custom_config = r'--oem 3 --psm 6'
    y_labels_text = pytesseract.image_to_string(sharpened_image, config=custom_config)
    pattern = re.compile(r'\b\d+\.\d+T\b')
    y_labels = pattern.findall(y_labels_text)

    return y_labels, sharpened_image, left_x, y_labels_text


def display_images(image, sharpened_image, graph_image, y_labels, padding_left=50):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image with Y-Axis Label Area Highlighted")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Processed Y-Axis Label Area")
    plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))

    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cv2.cvtColor(graph_image, cv2.COLOR_BGR2RGB))
    y_positions = np.linspace(padding_left, graph_image.shape[0] - padding_left, len(y_labels))
    for label, pos in zip(y_labels, y_positions):
        ax.text(-padding_left // 4, pos, label, color='black', fontsize=12, ha='right', va='center')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Graph with Updated Y-Axis Labels')
    plt.show()


def main(image_path):
    image = load_image(image_path)
    x, y, w, h = find_graph_contour(image)
    y_labels, sharpened_image, left_x, y_labels_text = extract_y_axis_labels(image, x, y, h)
    graph_image = image[y:y + h, x:x + w]

    if graph_image is None or graph_image.size == 0:
        raise ValueError("The graph image is empty or not correctly loaded.")

    print("Shape of graph_image after cropping:", graph_image.shape)
    print("Type of graph_image after cropping:", type(graph_image))

    display_images(image, sharpened_image, graph_image, y_labels)

    print("Extracted text from processed image:")
    print(y_labels_text)
    print("Extracted labels using regex:")
    print(y_labels)


if __name__ == "__main__":
    image_path = 'IXP.png'
    main(image_path)
