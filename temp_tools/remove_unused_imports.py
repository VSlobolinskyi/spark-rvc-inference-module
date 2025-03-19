import os
import subprocess


def process_file(file_path):
    """
    Run autoflake on the file to remove unused imports in-place.
    """
    try:
        # The --in-place option modifies the file, and
        # --remove-all-unused-imports removes unused imports.
        subprocess.run(
            ["autoflake", "--in-place", "--remove-all-unused-imports", file_path],
            check=True,
        )
        print(f"Processed: {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_path}: {e}")


def process_directory(root_dir):
    """
    Walk recursively through root_dir and process all .py files.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                process_file(file_path)


if __name__ == "__main__":
    # Start from the current directory (you can change this to your project root)
    project_root = os.getcwd()
    process_directory(project_root)
