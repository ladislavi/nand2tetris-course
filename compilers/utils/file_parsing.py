import os
import pathlib


def load_lines(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    return lines


def remove_whitespace(lines):
    """Remove whitespace and comments, leaving only valid instructions"""
    instructions = []
    for line in lines:
        if not line.startswith("/**"):
            cleaned = line.strip()
            line_parts = cleaned.split("//")
            if line_parts[0]:
                instructions.append(line_parts[0].strip())
    return instructions


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


def get_filename(filepath):
    return pathlib.Path(filepath).stem


def get_filepaths(filepath, suffix=".vm"):
    if not os.path.isdir(filepath):
        return [filepath]
    else:
        return [pth for pth in pathlib.Path(filepath).iterdir() if pth.suffix == suffix]
