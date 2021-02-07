import logging
import argparse
import re
from pathlib import Path
from utils import file_parsing

logger = logging.getLogger("assembler")
logger.setLevel(logging.INFO)
fh = logging.StreamHandler()
fh.setLevel(logging.INFO)
logger.addHandler(fh)


class Assembler:
    VAR_START_ADDRESS = 16

    COMP = {
        "0": "0101010",
        "1": "0111111",
        "-1": "0111010",
        "D": "0001100",
        "A": "0110000",
        "M": "1110000",
        "!D": "0001101",
        "!A": "0110001",
        "!M": "1110001",
        "-D": "0001111",
        "-A": "0110011",
        "-M": "1110011",
        "D+1": "0011111",
        "A+1": "0110111",
        "M+1": "1110111",
        "D-1": "0001110",
        "A-1": "0110010",
        "M-1": "1110010",
        "D+A": "0000010",
        "D+M": "1000010",
        "D-A": "0010011",
        "D-M": "1010011",
        "A-D": "0000111",
        "M-D": "1000111",
        "D&A": "0000000",
        "D&M": "1000000",
        "D|A": "0010101",
        "D|M": "1010101",
    }

    DEST = {
        "": "000",
        "M": "001",
        "D": "010",
        "MD": "011",
        "A": "100",
        "AM": "101",
        "AD": "110",
        "AMD": "111",
    }

    JUMP = {
        "": "000",
        "JGT": "001",
        "JEQ": "010",
        "JGE": "011",
        "JLT": "100",
        "JNE": "101",
        "JLE": "110",
        "JMP": "111",
    }

    def __init__(self):
        self.symbols = {
            "SP": 0,
            "LCL": 1,
            "ARG": 2,
            "THIS": 3,
            "THAT": 4,
            "R0": 0,
            "R1": 1,
            "R2": 2,
            "R3": 3,
            "R4": 4,
            "R5": 5,
            "R6": 6,
            "R7": 7,
            "R8": 8,
            "R9": 9,
            "R10": 10,
            "R11": 11,
            "R12": 12,
            "R13": 13,
            "R14": 14,
            "R15": 15,
            "SCREEN": 16384,
            "KBC": 24576,
        }
        self.instructions = []

    @staticmethod
    def _load_lines(filepath):
        with open(filepath) as f:
            lines = f.readlines()
        return lines

    @staticmethod
    def _remove_whitespace(lines):
        instructions = []
        for line in lines:
            cleaned = line.strip()
            line_parts = cleaned.split("//")
            if line_parts[0]:
                instructions.append(line_parts[0].strip())
        return instructions

    @staticmethod
    def _is_a(line):
        return line.startswith("@")

    def _read_instructions(self, filepath):
        self.instructions = file_parsing.parse_file(filepath)

    def _load_labels(self):
        """Add all labels to symbols lookup"""
        no_labels = []
        for line in self.instructions:
            if line.startswith("(") and line.endswith(")"):
                self.symbols[line.strip("()")] = len(no_labels)
            else:
                no_labels.append(line)

        self.instructions = no_labels

    def _load_variables(self):
        n_vars = 0
        for line in self.instructions:
            if self._is_a(line):
                addr = line[1:]
                if not addr.isdigit() and not addr in self.symbols:
                    var_addr = self.VAR_START_ADDRESS + n_vars
                    self.symbols[addr] = var_addr
                    n_vars += 1
                    logger.info(f"Adding var: '{addr}', at {var_addr}")

    def _replace_symbols(self, line):
        address = line[1:]
        if address in self.symbols:
            return f"@{self.symbols[address]}"
        else:
            return line

    def _translate_c_instruction(self, c_line):
        commands = re.split("[=;]", c_line)
        if "=" in c_line:
            dest = commands[0]
            comp = commands[1]
        else:
            dest = ""
            comp = commands[0]

        if ";" in c_line:
            jump = commands[-1]
        else:
            jump = ""

        logger.debug(f"Destination: {dest}, Computation: {comp}, Jump: {jump}")

        return f"111{self.COMP[comp]}{self.DEST[dest]}{self.JUMP[jump]}"

    def translate_line(self, line):
        if self._is_a(line):
            line = self._replace_symbols(line)
            return "0{0:015b}".format(int(line[1:]))
        else:
            return self._translate_c_instruction(line)

    def translate_file(self, filepath):
        logger.info("-" * 80)
        logger.info(f"Translating file: {filepath}")
        logger.info("-" * 80)
        self._read_instructions(filepath)
        self._load_labels()
        self._load_variables()
        machine_code = []
        for line in self.instructions:
            out_line = self.translate_line(line)
            logger.info(f"{line.ljust(15)}  >>  {out_line}")
            machine_code.append(self.translate_line(line))

        with open(f"out/{Path(filepath).stem}.hack", mode="wt") as f_out:
            f_out.write("\n".join(machine_code))


def main(filepath):
    assembler = Assembler()
    assembler.translate_file(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate HACK assembly language file into binary code"
    )
    parser.add_argument("filepath", type=str, help="Input file")
    args = parser.parse_args()
    main(**vars(args))
