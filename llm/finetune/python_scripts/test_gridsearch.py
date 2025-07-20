file_path = "/home/s2751141/dissertation/scottish_gaelic_chatbot/data/temp_data/english_test_set.txt"  # Update this path

try:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if lines:
        print("First entry:", lines[0].strip())
    else:
        print("File is empty.")

    print("File read complete.")

except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f" An error occurred: {e}")
