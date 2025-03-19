import os
import json

def load_mapping(mapping_path):
    """
    Load the mapping JSON file containing key-value pairs.
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    # Sort mapping items by length of the original text (key) in descending order
    sorted_mapping = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    return sorted_mapping

def process_file(file_path, sorted_mapping):
    """
    Read a file, replace occurrences of each key with its value, and write the file back
    if changes were made.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Skip binary files or files that can't be decoded as UTF-8
        print(f"Skipping binary or non-text file: {file_path}")
        return

    new_content = content
    # Replace longer keys first to avoid partial replacement of substrings.
    for original_text, translated_text in sorted_mapping:
        if original_text in new_content:
            new_content = new_content.replace(original_text, translated_text)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {file_path}")

def process_directory(root_dir, sorted_mapping, mapping_file_path):
    """
    Walk through all directories starting at root_dir and process each file.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Skip the mapping file itself.
            if os.path.abspath(file_path) == os.path.abspath(mapping_file_path):
                continue
            process_file(file_path, sorted_mapping)

if __name__ == '__main__':
    # Assume the script is run from the project root.
    project_root = os.getcwd()
    mapping_file_path = os.path.join(project_root, 'i18n', 'locale', 'en_US.json')
    
    if not os.path.exists(mapping_file_path):
        print(f"Mapping file not found: {mapping_file_path}")
    else:
        sorted_mapping = load_mapping(mapping_file_path)
        process_directory(project_root, sorted_mapping, mapping_file_path)
