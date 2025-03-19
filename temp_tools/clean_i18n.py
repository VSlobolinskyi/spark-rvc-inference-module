import os
import re

def process_file(file_path):
    """
    Reads a file, replaces occurrences of "Any text" (even if multiline)
    with just the "Any text" part, and writes the file back if changes are made.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping binary or non-text file: {file_path}")
        return

    # Regex pattern to match i18n( <whitespace> "Any text" <whitespace> )
    # It supports both single and double quoted strings, and uses DOTALL
    # so that the string literal may span multiple lines.
    pattern = re.compile(
        r'i18n\s*\(\s*(?P<quote>["\'])(?P<text>.*?)(?P=quote)\s*\)',
        re.DOTALL
    )

    # Replacement function returns only the string literal (including quotes).
    new_content = pattern.sub(lambda m: m.group("quote") + m.group("text") + m.group("quote"), content)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {file_path}")

def process_directory(root_dir):
    """
    Recursively traverse all files in root_dir and process them.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            process_file(file_path)

if __name__ == '__main__':
    # Start processing from the current working directory.
    project_root = os.getcwd()
    process_directory(project_root)
