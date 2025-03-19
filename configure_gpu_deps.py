#!/usr/bin/env python3
import sys
import re


def process_lines(lines, target_gpu):
    """
    Process lines from the pyproject.toml file.
    In the GPU configuration blocks, uncomment lines if they belong to the target block,
    and ensure lines in the non-target block are commented.
    """
    output_lines = []
    current_block = None  # None, "nvidia", or "amd"

    # Regex patterns for block markers and separator lines.
    nvidia_marker = re.compile(r"---\s*NVIDIA GPU configuration\s*---", re.IGNORECASE)
    amd_marker = re.compile(r"---\s*AMD GPU configuration\s*---", re.IGNORECASE)
    separator = re.compile(r"^#\s*-{5,}")  # a commented separator line

    for line in lines:
        # Check if this line marks the beginning of a GPU config block.
        if nvidia_marker.search(line):
            current_block = "nvidia"
            output_lines.append(line)
            continue
        elif amd_marker.search(line):
            current_block = "amd"
            output_lines.append(line)
            continue

        # If we hit a separator line, then end the current block.
        if separator.match(line):
            current_block = None
            output_lines.append(line)
            continue

        # Process lines within a GPU configuration block.
        if current_block is not None:
            # Remove newline to check content.
            stripped = line.rstrip("\n")
            if stripped.strip() == "":
                output_lines.append(line)
                continue

            # If this block is the target, ensure the line is uncommented.
            if current_block == target_gpu:
                # If the line is already uncommented, leave it.
                if not stripped.lstrip().startswith("#"):
                    output_lines.append(line)
                else:
                    # Remove the first occurrence of '#' with following space.
                    uncommented = re.sub(r"^(\s*)#\s?", r"\1", line)
                    output_lines.append(uncommented)
            else:
                # For the non-target block, ensure the line is commented.
                if stripped.lstrip().startswith("#"):
                    # Already commented, leave as-is.
                    output_lines.append(line)
                else:
                    # Add a '#' preserving the original indentation.
                    leading_space = re.match(r"^(\s*)", line).group(1)
                    output_lines.append(f"{leading_space}# {line.lstrip()}")
        else:
            # Outside of any GPU config block, just add the line.
            output_lines.append(line)
    return output_lines


def main():
    if len(sys.argv) != 2:
        print("Usage: python configure_gpu_deps.py <pyproject.toml> <gpu_type>")
        print("  where <gpu_type> is either 'nvidia' or 'amd'")
        sys.exit(1)

    gpu_type = sys.argv[1].lower()
    toml_path = "pyproject.toml"
    with open(toml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if gpu_type not in {"nvidia", "amd"}:
        print("gpu_type must be either 'nvidia' or 'amd'")
        sys.exit(1)

    with open(toml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = process_lines(lines, gpu_type)

    with open(toml_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"Updated {toml_path} for {gpu_type.upper()} GPU configuration.")


if __name__ == "__main__":
    main()
