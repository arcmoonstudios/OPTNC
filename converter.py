# root/converter.py
# Handles data conversion from various formats (JSON, text, PDF) into a suitable format for neural network processing.

import json
from PyPDF2 import PdfReader

class Converter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def convert(self, data):
        file_extension = self.input_path.split(".")[-1]
        if file_extension == "json":
            return self.convert_from_json(data)
        elif file_extension == "txt":
            return self.convert_from_text(data)
        elif file_extension == "pdf":
            return self.convert_from_pdf()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def convert_from_json(self, data):
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}")
    
    def convert_from_text(self, data):
        # Basic tokenization for demonstration
        return data.split()
    
    def convert_from_pdf(self):
        pdf_text = ""
        with open(self.input_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                pdf_text += page.extract_text()
        return self.convert_from_text(pdf_text)
