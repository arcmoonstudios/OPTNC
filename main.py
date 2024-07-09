# root/main.py
# Main entry point for the OmniPytron Neural Converter

import argparse
from converter import Converter
from data_processing.preprocessor import Preprocessor
from models.neural_model import NeuralModel
from utils.logger import setup_logger
from utils.file_handler import save_data, load_data
import numpy as np

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='OmniPytron Neural Converter')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to save the output file')
    parser.add_argument('--format', choices=['json', 'text', 'pdf'], default='json', help='Input format')
    args = parser.parse_args()

    try:
        # Initialize converter and load data
        converter = Converter(args.input_file, args.output_file)
        raw_data = load_data(args.input_file)
        
        # Convert and preprocess data
        converted_data = converter.convert(raw_data)
        preprocessor = Preprocessor()
        processed_data = preprocessor.process(converted_data)

        # Prepare data for the model (assuming binary classification for simplicity)
        X = processed_data[:, :-1]
        y = processed_data[:, -1]

        # Initialize and train the model
        input_dim = X.shape[1]
        model = NeuralModel(input_dim, 1)  # 1 output for binary classification
        model.train(X, y)

        # Save processed data
        save_data(processed_data, args.output_file)
        logger.info(f"Data successfully converted and saved to {args.output_file}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
