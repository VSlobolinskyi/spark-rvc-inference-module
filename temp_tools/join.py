#!/usr/bin/env python3
import subprocess
import os
import shutil


def clone_spark_repo(repo_url, clone_dir):
    """Clone the spark repo into a temporary directory."""
    print(f"Cloning {repo_url} into {clone_dir} ...")
    subprocess.run(["git", "clone", repo_url, clone_dir], check=True)
    print("Clone completed.")


def remove_git_directory(clone_dir):
    """Remove the .git directory from the cloned repo, adjusting permissions if necessary."""
    git_dir = os.path.join(clone_dir, ".git")
    if os.path.exists(git_dir):

        def handle_remove_error(func, path, exc_info):
            # Change the file to writable and try again.
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(git_dir, onerror=handle_remove_error)
        print(f"Removed .git directory from {clone_dir}")


def ensure_folder(folder_path):
    """Ensure that a folder exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")


def merge_spark_repo(clone_dir, spark_folder, temp_folder, duplicate_files):
    """Merge files from the cloned spark repo.

    - Files whose names are in duplicate_files and already exist in the root are moved to temp_folder.
    - All other items are moved to the spark_folder.
    """
    for item in os.listdir(clone_dir):
        source_path = os.path.join(clone_dir, item)
        # Determine the destination:
        root_dest = os.path.join(os.getcwd(), item)
        if item in duplicate_files and os.path.exists(root_dest):
            # Move duplicate file to TEMP folder
            dest_path = os.path.join(temp_folder, item)
            print(f"Duplicate '{item}' exists in root; moving it to {temp_folder}.")
        else:
            # Otherwise, place in the spark folder
            dest_path = os.path.join(spark_folder, item)
            print(f"Moving '{item}' to {spark_folder}.")
        shutil.move(source_path, dest_path)


def main():
    # Settings
    repo_url = "https://github.com/VSlobolinskyi/spark-tts-poetry.git"
    clone_dir = "spark_repo_temp"  # temporary folder for the cloned spark repo
    spark_folder = "spark"  # destination folder for spark-specific instruments
    temp_folder = "TEMP"  # destination folder for duplicate files
    # List of duplicate filenames that, if found in the root, will be moved to TEMP
    duplicate_files = {"pyproject.toml", ".gitignore", "LICENSE", "README.md"}

    # Clone the Spark repo
    clone_spark_repo(repo_url, clone_dir)

    # Remove the .git directory from the cloned repo
    remove_git_directory(clone_dir)

    # Ensure destination folders exist
    ensure_folder(spark_folder)
    ensure_folder(temp_folder)

    # Merge files from the cloned spark repo into the correct locations
    merge_spark_repo(clone_dir, spark_folder, temp_folder, duplicate_files)

    # Remove the temporary clone directory if it still exists (it should be empty now)
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
        print(f"Removed temporary directory: {clone_dir}")

    print("Spark repo merged successfully into the RVC project.")


if __name__ == "__main__":
    main()
