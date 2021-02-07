import os
import pathlib
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)

def load_lines(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    return lines

def to_file(contents, filepath, suffix='.vm'):
    """Write contents to file with the same name and new suffix"""
    filename = get_filename(filepath)
    out_filepath = f"out/{filename}{suffix}"
    with open(out_filepath, "w") as f:
        f.write(contents)
    logging.info(f"Output results to {out_filepath}")

def remove_whitespace(lines):
    """Remove whitespace and comments, leaving only valid instructions"""
    instructions = []
    for line in lines:
        cleaned = line.strip()
        if not (cleaned.startswith("/**") or cleaned.startswith("*")):
            line_parts = cleaned.split("//")
            if line_parts[0]:
                instructions.append(line_parts[0].strip())
    return instructions

def flatten_file(filepath):
    """Flatten .jack file into a single string"""
    jack_code = parse_file(filepath)
    code_out = []
    for line in jack_code:
        code_out += line.split(' ')

    return " ".join(code_out)


def strip_line(line):
    if not line.startswith("/**"):
        cleaned = line.strip()
        line_parts = cleaned.split("//")
        if line_parts[0]:
            yield line_parts[0].strip()


def parse_file(filepath):
    """Parse file into list of valid instructions"""
    lines = load_lines(filepath)
    return remove_whitespace(lines)

def clear_directory(filepath):
    pass


def get_filename(filepath):
    return pathlib.Path(filepath).stem


def get_filepaths(filepath, suffix=".vm"):
    if not os.path.isdir(filepath):
        return [filepath]
    else:
        return [pth for pth in pathlib.Path(filepath).iterdir() if pth.suffix == suffix]
