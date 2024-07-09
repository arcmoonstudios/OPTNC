# root/data_processing/image_graph_analyzer.py
# Implements tools for analyzing images of graphs

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import easyocr
from PIL import Image

class ImageGraphAnalyzer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])  # Initialize EasyOCR with English

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        return denoised

    def extract_text_from_image(self, image):
        result = self.reader.readtext(image)
        return ' '.join([text for _, text, _ in result])

    def detect_lines(self, image):
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        return lines

    def extract_color_palette(self, image, n_colors=5):
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        return colors.astype(int)

    def analyze_graph(self, image):
        preprocessed = self.preprocess_image(image)
        text = self.extract_text_from_image(preprocessed)
        lines = self.detect_lines(preprocessed)
        colors = self.extract_color_palette(image)
        
        if lines is not None and len(lines) > 10:
            graph_type = 'Line or Bar Graph'
        else:
            graph_type = 'Scatter Plot or Pie Chart'
        
        return {
            'graph_type': graph_type,
            'extracted_text': text,
            'num_lines_detected': len(lines) if lines is not None else 0,
            'color_palette': colors.tolist()
        }

    def plot_color_palette(self, colors):
        plt.figure(figsize=(10, 2))
        for i, color in enumerate(colors):
            plt.subplot(1, len(colors), i + 1)
            plt.axis('off')
            plt.imshow([[color]])
        plt.show()

if __name__ == '__main__':
    analyzer = ImageGraphAnalyzer()
    image_path = 'path_to_your_graph_image.jpg'
    image = analyzer.load_image(image_path)
    if image is not None:
        analysis_result = analyzer.analyze_graph(image)
        print('Graph Analysis Result:')
        print(f"Graph Type: {analysis_result['graph_type']}")
        print(f"Extracted Text: {analysis_result['extracted_text']}")
        print(f"Number of Lines Detected: {analysis_result['num_lines_detected']}")
        print('Color Palette:')
        analyzer.plot_color_palette(analysis_result['color_palette'])
    else:
        print(f'Failed to load image from {image_path}')
