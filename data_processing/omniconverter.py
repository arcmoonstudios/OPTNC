import json
import sys

class OmniConverter:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def log_error(self, error_message):
        try:
            with open(self.output_file.replace('.json', '_error.log'), 'a', encoding='utf-8') as f:
                f.write(f"ERROR: {error_message}\n")
        except Exception as e:
            print(f"Failed to log error: {e}")

    def analyze_data(self):
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = f.read()
            print(f"Read {len(data)} characters from input file.")
            structured_data = self.parse_input_text(data)
            return structured_data
        except Exception as e:
            error_message = f"Error during data analysis: {e}"
            print(error_message)
            self.log_error(error_message)
            return None

    def parse_input_text(self, text):
        try:
            sections = text.split('\n\n')
            structured_data = {}
            for i, section in enumerate(sections, start=1):
                subsections = section.split('\n')
                structured_data[i] = {}
                for j, subsection in enumerate(subsections):
                    structured_data[i][f'{chr(ord("a") + j)}'] = subsection.strip()
            return structured_data
        except Exception as e:
            error_message = f"Error during parsing input text: {e}"
            print(error_message)
            self.log_error(error_message)
            return {}

    def convert_to_json(self, data):
        try:
            result = {}
            for section, subsections in data.items():
                result.update(subsections)
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
            print(f"Data successfully written to {self.output_file}.")
        except Exception as e:
            error_message = f"Error during JSON conversion: {e}"
            print(error_message)
            self.log_error(error_message)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_optnc.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    converter = OmniConverter(input_file, output_file)
    structured_data = converter.analyze_data()
    if structured_data:
        converter.convert_to_json(structured_data)
        
        #      cd C:\Users\vxcor\source\repos\optnc.v01\data_processing\
        #      python omniconverter.py "C:\Users\vxcor\source\scripts\OPTNC\XoptncTarget.txt" "C:\Users\vxcor\source\scripts\OPTNC\XoptncResult.json"

