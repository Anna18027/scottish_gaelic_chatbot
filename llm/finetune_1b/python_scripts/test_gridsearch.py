# file_path = "/home/s2751141/dissertation/scottish_gaelic_chatbot/data/temp_data/english_test_set.txt"  # Update this path

# try:
#     with open(file_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     if lines:
#         print("First entry:", lines[0].strip())
#     else:
#         print("File is empty.")

#     print("File read complete.")

# except FileNotFoundError:
#     print(f"File not found: {file_path}")
# except Exception as e:
#     print(f" An error occurred: {e}")

import argparse
import os

def main():
    #add model results dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save results')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    #open data file
    # file_path = "/home/s2751141/dissertation/scottish_gaelic_chatbot/data/temp_data/english_test_set.txt"
    file_path = "/disk/scratch/s2751141/dissertation/scottish_gaelic_chatbot/data/english_test_set.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if lines:
            print("First entry:", lines[0].strip())
        else:
            print("File is empty.")

        print("File read complete.")

        test_output_file = os.path.join(output_dir, "results_ANNA.txt")
        with open(test_output_file, 'w') as f:
            f.write(lines[0].strip())

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f" An error occurred: {e}")

    print("Script complete.")

if __name__ == "__main__":
    main()
