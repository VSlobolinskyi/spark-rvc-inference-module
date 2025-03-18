#!/usr/bin/env python3
import sys
import re

def process_lines(lines, target_gpu):
    """
    Process the lines of a file and uncomment the configuration block corresponding
    to target_gpu ('nvidia' or 'amd') while leaving the other block commented.
    """
    output_lines = []
    current_block = None  # None, "nvidia", or "amd"
    
    # Define regexes to detect the markers
    nvidia_marker = re.compile(r'---\s*NVIDIA GPU configuration\s*---', re.IGNORECASE)
    amd_marker = re.compile(r'---\s*AMD GPU configuration\s*---', re.IGNORECASE)
    separator = re.compile(r'^#\s*-{5,}')  # a commented separator line (at least 5 dashes)
    
    for line in lines:
        stripped = line.lstrip()
        # Check for block start markers (they remain unchanged)
        if nvidia_marker.search(line):
            current_block = "nvidia"
            output_lines.append(line)
            continue
        elif amd_marker.search(line):
            current_block = "amd"
            output_lines.append(line)
            continue
        # End of block when encountering a separator line
        if separator.match(line):
            current_block = None
            output_lines.append(line)
            continue

        # If we're in a GPU configuration block and the line is commented, process it
        if current_block is not None and stripped.startswith("#"):
            # Remove the first '#' and any following space if we are in the target block.
            if current_block == target_gpu:
                # Uncomment by removing the first '#' (preserve indentation)
                # Using regex to remove a leading '#' with possible spaces
                uncommented = re.sub(r'^(?P<indent>\s*)#\s?', r'\g<indent>', line)
                output_lines.append(uncommented)
            else:
                # Leave the line commented for the non-target block
                output_lines.append(line)
        else:
            output_lines.append(line)
    return output_lines

def main():
    if len(sys.argv) != 3:
        print("Usage: python configure_gpu_deps.py <pyproject.toml> <gpu_type>")
        print("  where <gpu_type> is either 'nvidia' or 'amd'")
        sys.exit(1)

    toml_path = sys.argv[1]
    gpu_type = sys.argv[2].lower()
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
