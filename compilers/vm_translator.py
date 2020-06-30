import logging
import argparse
import utils.file_parsing as fp
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ],
)


class VmTranslator:
    def __init__(self):
        self.filename = None
        self.line_n = 0
        self.func_call_cnt = {}
        self.comp_call_cnt = {}

    _PUSH_TO_ST = ["@SP", "A=M", "M=D", "@SP", "M=M+1"]
    _POP_FROM_ST = ["@SP", "AM=M-1", "D=M"]
    _POINT_TO_ST = ["@SP", "A=M-1"]
    _POP_2_FROM_ST = _POP_FROM_ST + _POINT_TO_ST

    _PAIRWISE_CMDS = {
        "add": ["M=D+M"],
        "sub": ["M=M-D"],
        "and": ["M=D&M"],
        "or": ["M=D|M"],
    }

    _INVERT_CMDS = {"neg": ["M=-M"], "not": ["M=!M"]}

    _COMPARISON_CMDS = {"eq": "JEQ", "gt": "JGT", "lt": "JLT"}

    _LOGIC_CMDS = {**_PAIRWISE_CMDS, **_INVERT_CMDS, **_COMPARISON_CMDS}

    _BRANCH_CMDS = ("label", "goto", "if-goto")

    _FUNC_CMDS = {"function": 0, "call": 0, "return": 0}
    _FRAME_POINTERS = ["LCL", "ARG", "THIS", "THAT"]

    _BASE_POINTERS = {
        "local": "LCL",
        "argument": "ARG",
        "this": "THIS",
        "that": "THAT",
        "temp": 5,
    }

    def _parse_line(self, line):
        """Parse vm language command

        Returns:
             cmd_type, cmd, arg1, arg2
        """
        parts = line.split()
        if parts[0] in self._LOGIC_CMDS:
            return "logic", parts[0], None, None
        elif parts[0] in ["push", "pop"]:
            return "mem", parts[0], parts[1], parts[2]
        elif parts[0] in self._BRANCH_CMDS:
            return "branch", parts[0], parts[1], None
        elif parts[0] in self._FUNC_CMDS:
            cmd_type = "func"
            if len(parts) > 1:
                return cmd_type, parts[0], parts[1], parts[2]
            else:
                return cmd_type, parts[0], None, None
        else:
            raise ValueError(f"Command '{parts[0]}' not recognized")

    def translate_file_or_dir(self, filepath):
        asm_lines = []
        if os.path.isdir(filepath):
            out_file = f"{filepath}.asm"
            # only bootstrap dirs
            asm_lines += ["@256", "D=A", "@SP", "M=D"]
            asm_lines += self.translate_line("call Sys.init 0")
        else:
            out_file = f"{Path(filepath).stem}.asm"

        filepaths = fp.get_filepaths(filepath)
        for _file in filepaths:
            asm_lines += self.translate_file(_file)

        with open(out_file, mode="wt") as f_out:
            f_out.write("\n".join(asm_lines))
            logging.info(
                f"Output file: {f_out.name}, with {len(asm_lines)} of assembly code"
            )

    def translate_file(self, filepath):
        self.filename = fp.get_filename(filepath)
        vm_lines = fp.parse_file(filepath)
        logging.info(
            f"Translating input '{filepath}', with {len(vm_lines)} lines of VM code"
        )
        asm_lines = []
        for line in vm_lines:
            asm_lines += self.translate_line(line)

        return asm_lines

    def translate_line(self, line):
        cmd_type, cmd, arg1, arg2 = self._parse_line(line)
        assembly = [f"// {line}"]
        if cmd_type == "logic":
            assembly += self._translate_logic(cmd)
        elif cmd_type == "mem":
            assembly += self._translate_memory(cmd, arg1, arg2)
        elif cmd_type == "branch":
            assembly += self._translate_branching(cmd, arg1)
        elif cmd_type == "func":
            assembly += self._translate_function(cmd, arg1, arg2)
        else:
            raise ValueError(f"Command type '{cmd_type}' not recognized")
        self.line_n += 1

        return assembly

    def _translate_logic(self, cmd):
        if cmd in self._PAIRWISE_CMDS:
            return self._translate_pairwise(cmd)
        elif cmd in self._INVERT_CMDS:
            return self._translate_invert(cmd)
        elif cmd in self._COMPARISON_CMDS:
            return self._translate_comparison(cmd)
        else:
            raise ValueError(f"Command '{cmd}' not recognized")

    def _translate_memory(self, cmd, dest, idx):
        if cmd == "push":
            return self._translate_push(dest, idx)
        elif cmd == "pop":
            return self._translate_pop(dest, idx)
        else:
            raise ValueError(f"Command '{cmd}' not recognized")

    def _translate_branching(self, cmd, label):
        if cmd == "label":
            return [f"({label})"]
        elif cmd == "goto":
            return [f"@{label}", "0;JMP"]
        elif cmd == "if-goto":
            return self._POP_FROM_ST + [f"@{label}", "D;JNE"]
        else:
            raise ValueError(f"Command '{cmd}' not recognized")

    def _translate_function(self, cmd, f_name, n_args):
        if cmd == "function":
            return self._translate_declare(f_name, int(n_args))
        elif cmd == "call":
            return self._translate_call(f_name, int(n_args))
        elif cmd == "return":
            return self._translate_return()
        else:
            raise ValueError(f"Command '{cmd}' not recognized")

    def _save_frame(self):
        save_frame = []
        for pointer in self._FRAME_POINTERS:
            save_pointer = [f"@{pointer}", "D=M"] + self._PUSH_TO_ST
            save_frame += save_pointer

        return save_frame

    def _reposition_pointers(self, n_args):
        arg_pointer = [f"@{5+n_args}", "D=A", "@SP", "D=M-D", "@ARG", "M=D"]
        lcl_pointer = ["@SP", "D=M", "@LCL", "M=D"]
        return arg_pointer + lcl_pointer

    def _translate_call(self, f_name, n_args):
        if f_name not in self.func_call_cnt:
            self.func_call_cnt[f_name] = 0
        else:
            self.func_call_cnt[f_name] += 1
        return_addr = f"{f_name}.{self.func_call_cnt[f_name]}.return-addr"
        return [
            *[f"@{return_addr}", "D=A"],
            *self._PUSH_TO_ST,
            *self._save_frame(),
            *self._reposition_pointers(n_args),
            *[f"@{f_name}", "0;JMP"],
            *[f"({return_addr})"],
        ]

    def _translate_declare(self, f_name, n_vars):
        init_local = ["@0", "D=A", *self._PUSH_TO_ST] * n_vars
        return [f"({f_name})"] + init_local

    def _save_return_tmp(self, frame, ret):
        return [
            "@LCL",
            "D=M",
            f"@{frame}",
            "M=D",
            "@5",
            "D=A",
            f"@{frame}",
            "A=M-D",
            "D=M",
            f"@{ret}",
            "M=D",
        ]

    def _restore_pointers(self, frame):
        assembly = []
        for idx, pointer in enumerate(self._FRAME_POINTERS[::-1]):
            assembly += [
                f"@{idx+1}",
                "D=A",
                f"@{frame}",
                "A=M-D",
                "D=M",
                f"@{pointer}",
                "M=D",
            ]
        return assembly

    def _translate_return(self):
        frame = f"tmp-frame"
        ret = f"tmp-ret"
        return [
            *self._save_return_tmp(frame, ret),
            *self._POP_FROM_ST,
            *["@ARG", "A=M", "M=D"],
            *["@ARG", "D=M+1", "@SP", "M=D"],
            *self._restore_pointers(frame),
            *[f"@{ret}", "A=M", "0;JMP"],
        ]

    def _translate_pairwise(self, cmd):
        return self._POP_2_FROM_ST + self._PAIRWISE_CMDS[cmd]

    def _translate_invert(self, cmd):
        return self._POINT_TO_ST + self._INVERT_CMDS[cmd]

    def _translate_comparison(self, cmd):
        if cmd not in self.comp_call_cnt:
            self.comp_call_cnt[cmd] = 0
        else:
            self.comp_call_cnt[cmd] += 1

        return self._POP_2_FROM_ST + [
            "D=M-D",
            f"@CONDITION_TRUE.{self.comp_call_cnt[cmd]}",
            f"D;{self._COMPARISON_CMDS[cmd]}",
            "@0",
            "D=A",
            f"@PUSH_TO_STACK.{self.comp_call_cnt[cmd]}",
            "0;JMP",
            f"(CONDITION_TRUE.{self.comp_call_cnt[cmd]})",
            "@1",
            "D=-A",
            f"(PUSH_TO_STACK.{self.comp_call_cnt[cmd]})",
            "@SP",
            "A=M-1",
            "M=D",
        ]

    def _translate_push(self, src, idx):
        if src == "constant":
            get_val = ["D=A"]
        else:
            get_val = ["D=M"]
        return [
            *self._store_addr(src, idx),
            *self._point_to_addr(src, idx),
            *get_val,
            *self._PUSH_TO_ST,
        ]

    def _translate_pop(self, dest, idx):
        return [
            *self._store_addr(dest, idx),
            *self._POP_FROM_ST,
            *self._point_to_addr(dest, idx),
            *["M=D"],
        ]

    def _store_addr(self, dest, idx):
        if dest in self._BASE_POINTERS.keys():
            return [
                f"@{idx}",
                "D=A",
                f"@{self._BASE_POINTERS[dest]}",
                f"D={'M' if dest != 'temp' else 'A'}+D",
                f"@tmp-addr",
                "M=D",
            ]
        else:
            return []

    def _point_to_addr(self, dest, idx):
        if dest in self._BASE_POINTERS.keys():
            return [f"@tmp-addr", "A=M"]
        elif dest == "constant":
            return [f"@{idx}"]
        elif dest == "static":
            return [f"@{self.filename}.{idx}"]
        elif dest == "pointer":
            if idx == "0":
                return ["@THIS"]
            elif idx == "1":
                return ["@THAT"]
            else:
                raise ValueError(f"Invalid index for '{dest}': '{idx}', 0/1 expected")
        else:
            raise ValueError(f"Invalid segment specified '{dest}'")


def main(filepath):
    translator = VmTranslator()
    translator.translate_file_or_dir(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate VM code to HACK assembly language"
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Input file or directory (files must have .vm extension)",
    )
    args = parser.parse_args()
    main(**vars(args))
