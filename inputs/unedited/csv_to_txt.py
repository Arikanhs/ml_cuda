import csv
import os
import sys

def csv_to_txt(input_file, output_file):
    try:
        with open(input_file, 'r') as csvfile, open(output_file, 'w') as txtfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) == 2:
                    txtfile.write(f"{row[0]} {row[1]}\n")
        print(f"Conversion complete. Output saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script_name.py input.csv [output.txt]")
        print("If output filename is not provided, it will use input filename with .txt extension")
        return

    input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = os.path.splitext(input_file)[0] + '.txt'

    csv_to_txt(input_file, output_file)

if __name__ == "__main__":
    main()