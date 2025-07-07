import argparse
import os

import torch_xla.core.xla_builder as xb


def find_pb_file(directory):
    """
    Find a .pb file in the specified directory.
    Returns the full path to the file if found, None otherwise.
    """
    # List all files in the directory
    try:
        files = os.listdir(directory)
        # Filter for .pb files
        pb_files = [f for f in files if f.endswith(".pb")]

        if not pb_files:
            return None
        if len(pb_files) > 1:
            print(f"Warning: Multiple .pb files found in {directory}. Using {pb_files[0]}")

        # Return the full path to the first .pb file
        return os.path.join(directory, pb_files[0])
    except OSError as e:
        print(f"Error accessing directory: {str(e)}")
        return None


def get_output_name(input_path, output_name=None):
    """
    Generate an output filename based on the input path or specified name.
    Returns a string with .txt extension.
    """
    if output_name:
        # If output name is provided, use it
        base_name = output_name
    else:
        abs_path = os.path.abspath(input_path)
        base_name = os.path.basename(abs_path)

    # Ensure the base name ends with .txt
    if not base_name.endswith(".hlo"):
        base_name += ".hlo"

    return base_name


def main():
    # Set up argument parser with more detailed help messages
    parser = argparse.ArgumentParser(
        description="Process an HLO module proto file and save its computation to a text file."
    )

    # Add arguments with detailed help messages
    parser.add_argument("input_path", type=str, help="Directory containing a .pb file, or direct path to a .pb file")
    parser.add_argument(
        "--output", "-o", type=str, help="Optional output filename (default: directory name + .txt)", default=None
    )

    args = parser.parse_args()

    # Determine if the input path is a directory or file
    input_path = args.input_path
    if os.path.isdir(input_path):
        # Find .pb file in directory
        pb_file = find_pb_file(input_path)
        if not pb_file:
            print(f"Error: No .pb file found in directory '{input_path}'")
            return
    else:
        # Use the provided path directly
        pb_file = input_path
        if not pb_file.endswith(".pb"):
            print(f"Error: File '{pb_file}' is not a .pb file")
            return
        if not os.path.exists(pb_file):
            print(f"Error: File '{pb_file}' does not exist")
            return

    # Generate output filename
    output_file = get_output_name(input_path, args.output)

    try:
        # Read and process the .pb file
        with open(pb_file, mode="rb") as f:
            comp = xb.computation_from_module_proto("foo", f.read())

            # Get the HLO computation
            hlo_output = xb.get_computation_hlo(comp)

            # Write the output to a file
            with open(output_file, "w") as out_f:
                out_f.write(hlo_output)

            print(f"Successfully wrote HLO computation to '{output_file}'")

    except Exception as e:
        print(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    main()
