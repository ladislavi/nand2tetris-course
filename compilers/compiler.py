import logging
import argparse
import utils.file_parsing as fp
import re
from collections import OrderedDict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ],
)


class JackTokenizer(object):

    _KEYWORDS = {
        "class",
        "constructor",
        "function",
        "method",
        "field",
        "static",
        "var",
        "int",
        "char",
        "boolean",
        "void",
        "true",
        "false",
        "null",
        "this",
        "let",
        "do",
        "if",
        "else",
        "while",
        "return",
    }

    _SYMBOLS = "{}()[\]\.,;\+-\*\/&\|<>=~"
    _IDENTIFIER_RE = "^[a-zA-Z_][a-zA-Z0-9_]+"
    _STRING_CONSTANT_RE = "^(\".*\")"
    _INT_CONSTANT_RE = "^[0-9]+"
    _MIN_INT = 0
    _MAX_INT = 32767

    def _is_symbol_or_whitespace(self, char):
        if char in self._SYMBOLS + " ":
            return True
        else:
            return False

    def _get_single_token(self, input):
        idx = 0
        if self._is_symbol_or_whitespace(input[idx]):
            return input[idx]
        elif input[idx] == '"':
            sr = re.search(self._STRING_CONSTANT_RE, input)
            return sr.group(1)

        while True:
            idx += 1
            if self._is_symbol_or_whitespace(input[idx]):
                return input[:idx]

    def _tokenize(self, input):
        tokens = []
        while input:
            token = self._get_single_token(input)
            if not token == " ":
                tokens.append(token)
            input = input[len(token) :]

        return tokens

    def _analyze_token(self, token):
        if token in self._SYMBOLS:
            return token, 'symbol'
        elif token in self._KEYWORDS:
            return token, 'keyword'
        elif re.match(self._STRING_CONSTANT_RE, token):
            return token[1:-1], 'stringConst'
        elif re.match(self._INT_CONSTANT_RE, token):
            if self._MIN_INT <= int(token) <= self._MAX_INT:
                return token, 'intConst'
            else:
                raise ValueError(f"Integer out of bounds ({self._MIN_INT}..{self._MAX_INT}): {token}")
        elif re.match(self._IDENTIFIER_RE, token):
            return token, 'identifier'
        else:
            raise ValueError(f"Unknown token type: '{token}'")

    def tokenize(self, input):
        tokens = self._tokenize(input)
        return [self._analyze_token(token) for token in tokens]

class JackAnalyzer(object):

    _KEYWORDS = {
        "class",
        "constructor",
        "function",
        "method",
        "field",
        "static",
        "var",
        "int",
        "char",
        "boolean",
        "void",
        "true",
        "false",
        "null",
        "this",
        "let",
        "do",
        "if",
        "else",
        "while",
        "return",
    }

    _SYMBOLS = "[{}()[].,;+-*/&|<>=~]"
    _TERMINATOR = ";"
    _MIN_INT = 0
    _MAX_INT = 32767
    _IDENTIFIER_RE = "[a-zA-Z_][a-zA-Z0-9_]*"

    _CLASS_VARS_RE = "static|field"
    _SUBROUTINE_TYPES_RE = "constructor|function|method"
    _STATEMENT_RE = "let|if|while|do|return"
    _OP_RE = "\+|-|\*|\/|&|\||<|>|="
    _UNARY_OP_RE = "-|~"
    _INTEGER_CONSTANT_RE = "[0-9]*"
    _STRING_CONSTANT_RE = "\"(.*)\""

    _BASE_TYPES = {"int", "char", "boolean"}

    def __init__(self):
        self.class_names = set()
        self.class_var_names = set()
        self.var_names = set()
        self.class_structure = None

    @property
    def types(self):
        return self._BASE_TYPES | self.class_names

    def compile_file_or_dir(self, filepath):
        filepaths = fp.get_filepaths(filepath, ".jack")
        for _file in filepaths:
            self.compile_file(_file)

    @staticmethod
    def _strip_line(line):
        out_line = ""
        if not line.startswith("/**"):
            cleaned = line.strip()
            line_parts = cleaned.split("//")
            if line_parts[0]:
                out_line = line_parts[0].strip()
        return out_line

    def _flatten_file(self, fileobj):
        out = ""
        for line in fileobj:
            line = self._strip_line(line)
            if line:
                out += f"{line} "

        return out

    def _is_identifier(self, text):
        return bool(re.match(self._IDENTIFIER_RE, text))

    def _analyze_class(self, code):
        class_re = r"class\s*(" + self._IDENTIFIER_RE + ")\s*\{(.*)\}"
        if re.match(class_re, code):
            sr = re.search(class_re, code)
            code = sr.group(3)
            class_declaration = (
                ("keyword", "class"),
                ("identifier", sr.group(2)),
                ("symbol", "{"),
            )

            code, class_var_dec = self._analyze_class_var_dec(code)
            subroutine_dec = self._analyze_subroutine_dec(code)

            if class_var_dec:
                class_declaration += class_var_dec
            if subroutine_dec:
                class_declaration += subroutine_dec

            class_declaration += (("symbol", "}"),)
            return "class", class_declaration
        else:
            raise ValueError(f"Code is not a valid class: \n{code}")

    def _is_kwd_or_idf(self, token):
        if token in self._KEYWORDS:
            return "keyword"
        elif re.match(self._IDENTIFIER_RE, token):
            return "identifier"
        else:
            raise ValueError(f"Token '{token}' is neither keyword or identifier")

    def _parse_var_names(self, code):
        var_names = code.split(",")
        vars_out = ()
        for idx, var in enumerate(var_names):
            if self._is_identifier(var):
                vars_out += (("identifier", var),)
            else:
                raise ValueError(f"Invalid variable name: '{var}'")
            if idx != len(var_names):
                vars_out += (("symbol", ","),)

        return vars_out

    def _analyze_class_var_dec(self, code):
        class_var_re = (
            r"("
            + self._CLASS_VARS_RE
            + r")\s*("
            + self._IDENTIFIER_RE
            + r")\s*(.*?);(.*)"
        )
        class_vars = ()
        while re.match(class_var_re, code):
            sr = re.search(class_var_re, code)
            code = sr.group(4)
            var_dec = ()
            var_dec += (("keyword", sr.group(1)),)
            var_dec += ((self._is_kwd_or_idf(sr.group(2)), sr.group(2)),)
            var_dec += self._parse_var_names(sr.group(3))
            class_vars += (("classVarDec", var_dec),)
        return code, class_vars

    def _analyze_subroutine_dec(self, code):
        subroutine_dec_re = (
            r"("
            + self._SUBROUTINE_TYPES_RE
            + r")\s*("
            + self._IDENTIFIER_RE
            + r")\s*("
            + self._IDENTIFIER_RE
            + r")\s*\((.*?)\)\s*\{(.*?)\}\s*(.*)"
        )

        subroutines = ()
        while re.match(subroutine_dec_re, code):
            subroutine_dec = ()
            sr = re.search(subroutine_dec_re, code)
            subroutine_dec += (("keyword", sr.group(1)),)  # subroutine type
            subroutine_dec += (
                (self._is_kwd_or_idf(sr.group(2)), sr.group(2)),
            )  # subroutine type
            subroutine_dec += (("identifier", sr.group(3)),)  # subroutine name
            subroutine_dec += (
                ("parameterList", self._analyze_parameter_list(sr.group(4))),
            )  # param list
            subroutine_dec += (
                ("subroutineBody", self._analyze_subroutine_body(sr.group(5))),
            )  # function body

            subroutines += subroutine_dec
            code = sr.group(6)
        return subroutines

    def _analyze_parameter_list(self, code):
        param_list = ()
        params = code.split(",")
        for idx, param in enumerate(params):
            type, name = param.split()
            if self._is_identifier(name):
                param_list += (self._is_kwd_or_idf(type), name)
            else:
                raise ValueError(
                    f"Parameter name must be valid identifier, '{name}' provided"
                )
            if idx != len(params):
                param_list += (("symbol", ","),)
        return param_list

    def _analyze_subroutine_body(self, code):
        subroutine_body = (("symbol", "{"),)
        while code:
            code, var_declarations = self._analyze_var_declaration(code)
            code, statements = self._analyze_statements(code)

            if var_declarations:
                subroutine_body += var_declarations
            if statements:
                subroutine_body += statements

        return (subroutine_body + ("symbol", "}"),)

    def _analyze_var_declaration(self, code):
        var_dec_re = r"var\s*(" + self._IDENTIFIER_RE + r")\s*(.*?);(.*)"
        var_declarations = ()
        while re.match(var_dec_re, code):
            sr = re.search(var_dec_re, code)
            var_declaration = ()
            var_declaration += (("keyword", "var"),)
            type = sr.group(1)
            var_declaration += ((self._is_kwd_or_idf(type), type),)
            var_declaration += self._parse_var_names(sr.group(2))

            var_declarations += (("varDec", var_declaration),)
            code = sr.group(3)
        return code, var_declarations

    def _analyze_statements(self, code):
        statement_re = "\s*(" + self._STATEMENT_RE + ")\s.*"
        statements = ()
        while True:
            sr = re.search(statement_re, code)
            if sr:
                if sr.group(1) == "if":
                    code, statement = self._analyze_if(code)
                elif sr.group(1) == "let":
                    code, statement = self._analyze_let(code)
                elif sr.group(1) == "while":
                    code, statement = self._analyze_while(code)
                elif sr.group(1) == "do":
                    code, statement = self._analyze_do(code)
                elif sr.group(1) == "return":
                    code, statement = self._analyze_return(code)
                else:
                    raise ValueError(f"Can't parse statement: '{sr.group(1)}'")

                statements += statement
            else:
                break

        return statements

    def _analyze_do(self, code):
        do_re = "do\s*(.*?);"
        sr = re.search(do_re, code)
        do_statement = ()
        if sr:
            do_statement += (("keyword", "do"),)
            do_statement += self._analyze_subroutine_call(sr.group(1))
            do_statement += (("symbol", ";"),)
        return code, do_statement

    def _analyze_subroutine_call(self, code):
        subroutine_re = (
            "("
            + self._IDENTIFIER_RE
            + ")\s*("
            + self._IDENTIFIER_RE
            + "|"
            + self._IDENTIFIER_RE
            + "\."
            + self._IDENTIFIER_RE
            + ")\((.*)\)"
        )

        sr = re.search(subroutine_re, code)
        subroutine_call = ()
        if sr:
            subroutine_name = sr.group(1)
            if len(subroutine_name) > 1:
                subroutine_call += (("identifier", subroutine_name[0]),)
                subroutine_call += (("symbol", "."),)
                subroutine_call += (("identifier", subroutine_name[1]),)
            else:
                subroutine_call += (("identifier", subroutine_name[0]),)

            subroutine_call += (self._analyze_expression_list(sr.group(2)),)
        return code, subroutine_call

    def _analyze_expression_list(self, code):
        pass

    def _analyze_expression(self, code):
        # if is_nested (= has "()"):
        # _analyze_expression(nested_part)
        # elif has_unary_op:
        # unary_op + _

        pass

    def _analyze_unary_op(self, code):
        pass

    def _analyze_term(self, code):
        if self._is_constant(code):
            return self._analyze_constants(code)
        pass

    def _is_constant(self, code):
        if (
            re.match(self._STRING_CONSTANT_RE, code)
            or re.match(self._INTEGER_CONSTANT_RE, code)
            or code in self._KEYWORDS
        ):
            return True
        else:
            return False

    def _analyze_constants(self, code):
        if code in self._KEYWORDS:
            return "", (("keyword", code),)
        elif re.match(self._INTEGER_CONSTANT_RE, code):
            if self._MIN_INT <= int(code) and int(code) <= self._MAX_INT:
                return "", (("integerConstant", code),)
            else:
                raise ValueError(f"Value '{code}' out of bounds for 16bit int")
        elif re.match(self._STRING_CONSTANT_RE, code):
            return "", (("stringConstant", code),)
        else:
            raise ValueError(f"Code '{code}' failed to parse as constant")

    def compile_file(self, filepath):
        out_filepath = f"{Path(filepath).stem}.vm"
        with open(out_filepath, mode="w") as out_file:
            with open(filepath, mode="r", encoding="utf-8") as in_file:
                print(self._flatten_file(in_file))
                # self.write_xml(out_file, token)


def main(filepath):
    # compiler = JackCompiler()
    # compiler.compile_file_or_dir(filepath)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Jack code into VM code")
    parser.add_argument(
        "filepath",
        type=str,
        help="Input file or directory (files must have .jack extension)",
    )
    args = parser.parse_args()
    main(**vars(args))
