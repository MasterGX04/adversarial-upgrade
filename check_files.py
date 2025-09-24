import os
import re
from collections import Counter

# --- Configuration ---
# The directory you want to scan.
TARGET_DIR = r"D:\Games\FF7 Rebirth Download"
# The text that must be in the filename.
FILENAME_SUBSTRING = "Monkey_Setup"
# The required file extension.
FILE_EXTENSION = ".rar"
# The total number of files expected in the sequence.
EXPECTED_TOTAL = 251

def check_file_integrity(directory, substring, extension, max_number):
    """
    Scans a directory to verify a sequence of numbered .rar files.

    Args:
        directory (str): The path to the directory to check.
        substring (str): A string that must be present in the filenames.
        extension (str): The required file extension (e.g., ".rar").
        max_number (int): The highest number expected in the file sequence.
    """
    print(f"🔍 Starting scan in directory: {directory}\n")
    
    if not os.path.isdir(directory):
        print(f"Error: Directory not found in {directory}")
        return

    found_numbers = []
    incorrect_files = []
    
    pattern = re.compile(r'\.part(\d+)\.rar$', re.IGNORECASE)
    
    for filename in os.listdir(directory):
        if substring in filename:
            if not filename.lower().endswith(extension):
                incorrect_files.append(filename)
                continue;
            
            # Try to match pattern and extract number
            match = pattern.search(filename)
            if match:
                number = int(match.group(1))
                found_numbers.append(number)
            else:
                incorrect_files.append(filename)
                
    number_counts = Counter(found_numbers)
    duplicates = {num for num, count in number_counts.items() if count > 1}
    
    expected_set = set(range(1, max_number + 1))
    found_set = set(found_numbers)
    missing = sorted(list(expected_set - found_set))
    
    print("--- Scan Report ---\n")
    
    if not found_numbers and not incorrect_files:
        print("🤷 No files containing '{substring}' were found.")
        return

    # Report on duplicates
    if duplicates:
        print(f"🚨 Found {len(duplicates)} duplicate file number(s):")
        print(f"   {sorted(list(duplicates))}\n")
    else:
        print("✅ No duplicate file numbers found.\n")

    # Report on missing files
    if missing:
        print(f"❗ Found {len(missing)} missing file number(s) from 1 to {max_number}:")
        print(f"   {missing}\n")
    else:
        print(f"✅ All numbers from 1-{max_number} are present.\n")
        
    # Report on files that matched the substring but not the pattern or extension
    if incorrect_files:
        print(f"⚠️ Found {len(incorrect_files)} file(s) with incorrect format/name:")
        for f in incorrect_files:
            print(f"   - {f}")
        print()

    # Final summary
    print(found_set)
    total_found = len(found_set)
    if not duplicates and not missing and not incorrect_files and total_found == max_number:
        print("🎉 Success! All files are present and correctly named.")
    else:
        print(f"📊 Summary: Found {total_found} unique and valid files out of {max_number} expected.")
        
if __name__ == "__main__":
    check_file_integrity(TARGET_DIR, FILENAME_SUBSTRING, FILE_EXTENSION, EXPECTED_TOTAL)