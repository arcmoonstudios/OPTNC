import json
import sys

class OmniConverter:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def log_and_print_error(self, error_message):
        """Log and print error messages."""
        try:
            with open(self.output_file.replace('.json', '_error.log'), 'a', encoding='utf-8') as f:
                f.write(f"ERROR: {error_message}\n")
        except Exception as e:
            print(f"Failed to log error: {e}")
        print(error_message)

    def read_input_file(self):
        """Read data from the input file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = f.read()
            print(f"Read {len(data)} characters from input file.")
            return data
        except Exception as e:
            self.log_and_print_error(f"Error reading input file: {e}")
            return None

    def parse_input_text(self, text):
        """Parse the input text into a structured format."""
        try:
            sections = text.split('\n\n')
            structured_data = {i+1: {f'{chr(ord("a") + j)}': subsection.strip() 
                                     for j, subsection in enumerate(section.split('\n'))} 
                               for i, section in enumerate(sections)}
            return structured_data
        except Exception as e:
            self.log_and_print_error(f"Error during parsing input text: {e}")
            return {}

    def write_to_json(self, data):
        """Write the structured data to a JSON file."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"Data successfully written to {self.output_file}.")
        except Exception as e:
            self.log_and_print_error(f"Error during JSON conversion: {e}")

    def analyze_and_convert(self):
        """Read, parse, and convert data to JSON."""
        data = self.read_input_file()
        if data:
            structured_data = self.parse_input_text(data)
            self.write_to_json(structured_data)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_optnc.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    converter = OmniConverter(input_file, output_file)
    converter.analyze_and_convert()
