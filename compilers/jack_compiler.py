import logging
import argparse
import utils.file_utils as fu
import re
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)


def trim_children(parent_node):
    """Removes empty entries (symbols) in parse tree"""
    out = []
    for child in parent_node.val:
        if isinstance(child, TerminalToken):
            out.append(child)
        elif child:
            out.append(trim_children(child))
    return Token(parent_node.label, out)


class Token(object):
    def __init__(self, label, val):
        self.label = label
        self.val = val

    def __str__(self):
        return f"Token({self.label}, '{self.val}')"


class TerminalToken(Token):
    def __str__(self):
        return f"TerminalToken({self.label}, '{self.val}')"


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

    _MIN_INT = 0
    _MAX_INT = 32767

    _IDENTIFIER_RE = "^[a-zA-Z_]+[a-zA-Z0-9_]*"
    _STRING_CONSTANT_RE = "^(\".*?\")"
    _INT_CONSTANT_RE = "^[0-9]+"

    def __init__(self):
        self.tokens = []

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

    def _is_symbol_or_whitespace(self, char):
        return char in self._SYMBOLS + " "

    def _analyze_token(self, token):
        if token in self._SYMBOLS:
            return TerminalToken('symbol', token)
        elif token in self._KEYWORDS:
            return TerminalToken('keyword', token)
        elif re.match(self._STRING_CONSTANT_RE, token):
            return TerminalToken('stringConstant', token[1:-1])
        elif re.match(self._INT_CONSTANT_RE, token):
            if self._MIN_INT <= int(token) <= self._MAX_INT:
                return TerminalToken('integerConstant', token)
            else:
                raise ValueError(
                    f"Integer out of bounds ({self._MIN_INT}..{self._MAX_INT}): {token}"
                )
        elif re.match(self._IDENTIFIER_RE, token):
            return TerminalToken('identifier', token)
        else:
            raise ValueError(f"Unknown token type: '{token}'")

    def _tokenize(self, input):
        while input:
            token = self._get_single_token(input)
            if not token == " ":
                logging.debug(f"Processing token: '{token}'")
                self.tokens.append(self._analyze_token(token))
            input = input[len(token) :]

        return self.tokens

    def tokenize_file(self, filepath):
        logging.info(f"Tokenizing {filepath}")
        jack_code = fu.flatten_file(filepath)
        tokens = self._tokenize(jack_code)
        logging.info(f"Tokenizing done. {filepath} tokenized into {len(tokens)}")
        return tokens


class JackAnalyzer(object):

    _TERMINALS = {
        'keyword',
        'symbol',
        'integerConstant',
        'stringConstant',
        'identifier',
    }

    _OPS = {'+', '-', '*', '/', '&', '|', '<', '>', '='}
    _UNARY_OPS = {'-', '~'}
    _KEYWORD_CONSTS = {'true', 'false', 'null', 'this'}
    _CLASS_VAR_TYPES = {'static', 'field'}
    _SUBROUTINE_TYPES = {'constructor', 'function', 'method'}
    _STATEMENTS = {'let', 'if', 'while', 'do', 'return'}

    def __init__(self):
        self.tokens = []
        self.parse_tree = None
        self.idx = 0

    def _check_token_val(self, expected, offset=0):
        """Returns true if token matches values"""
        if isinstance(expected, str):
            expected = {expected}
        val = self.tokens[self.idx + offset].val
        return val in expected

    def _check_token_type(self, expected, offset=0):
        """Returns true if token matches values"""
        if isinstance(expected, str):
            expected = {expected}
        return self.tokens[self.idx + offset].label in expected

    def _get_terminal_token(self, write=True):
        """Returns terminal token and moves index"""
        out = self.tokens[self.idx]
        self.idx += 1
        if write:
            return out

    def _validate_token_type(self, expected):
        """Raises an error if token doesn't match expected type"""
        if not self._check_token_type(expected):
            raise ValueError(
                f"Expected '{expected}', got {self.tokens[self.idx]} at {self.idx}"
            )

    def _validate_token_val(self, expected):
        """Raises an error if token doesn't match expected values"""
        if not self._check_token_val(expected):
            raise ValueError(
                f"Expected '{expected}', got {self.tokens[self.idx]} at {self.idx}"
            )

    def _parse_identifier(self):
        self._validate_token_type('identifier')
        return self._get_terminal_token()

    def _parse_keyword(self, keywords=None):
        self._validate_token_type('keyword')
        if keywords:
            self._validate_token_val(keywords)
        return self._get_terminal_token()

    def _parse_symbol(self, values=None):
        self._validate_token_type('symbol')
        if values:
            self._validate_token_val(values)
        return self._get_terminal_token(False)

    def _parse_op(self, values=None):
        self._validate_token_type('symbol')
        if values:
            self._validate_token_val(values)
        return TerminalToken('op', self._get_terminal_token().val)

    def _parse_int_const(self):
        self._validate_token_type('integerConstant')
        return self._get_terminal_token()

    def _parse_str_const(self):
        self._validate_token_type('stringConstant')
        return self._get_terminal_token()

    def _parse_type(self, extend=None):
        keywords = ['int', 'char', 'boolean']
        if extend:
            keywords.append(extend)

        if self._check_token_val(keywords):
            return self._parse_keyword()
        else:
            return self._parse_identifier()

    def _parse_class(self):
        class_def = [
            self._parse_keyword('class'),
            self._parse_identifier(),
            self._parse_symbol('{'),
        ]

        while self._check_token_val(self._CLASS_VAR_TYPES):
            class_def.append(self._parse_class_var_dec())

        while self._check_token_val(self._SUBROUTINE_TYPES):
            class_def.append(self._parse_subroutine_dec())

        class_def.append(self._parse_symbol('}'))

        self.parse_tree = Token('class', class_def)
        return self.parse_tree

    def _parse_class_var_dec(self):
        class_var_dec = [
            self._parse_keyword(),
            self._parse_type(),
            self._parse_identifier(),
        ]

        while self._check_token_val(','):
            class_var_dec.append(self._parse_symbol(','))
            class_var_dec.append(self._parse_identifier())

        class_var_dec.append(self._parse_symbol(';'))

        return Token('classVarDec', class_var_dec)

    def _parse_subroutine_dec(self):
        return Token(
            'subroutineDec',
            [
                self._parse_keyword(),
                self._parse_type('void'),
                self._parse_identifier(),
                self._parse_symbol('('),
                self._parse_parameter_list(),
                self._parse_symbol(')'),
                self._parse_subroutine_body(),
            ],
        )

    def _parse_parameter_list(self):
        param_list = []
        if not self._check_token_val(')'):
            param_list = [self._parse_type(), self._parse_identifier()]

            while self._check_token_val(','):
                param_list.append(self._parse_symbol(','))
                param_list.append(self._parse_type())
                param_list.append(self._parse_identifier())

        return Token('parameterList', param_list)

    def _parse_subroutine_body(self):
        subroutine_body = [self._parse_symbol('{')]

        while self._check_token_val('var'):
            subroutine_body.append(self._parse_var_dec())

        subroutine_body.append(self._parse_statements())
        subroutine_body.append(self._parse_symbol('}'))

        return Token('subroutineBody', subroutine_body)

    def _parse_var_dec(self):
        var_dec = [
            self._parse_keyword('var'),
            self._parse_type(),
            self._parse_identifier(),
        ]

        while self._check_token_val(','):
            var_dec.append(self._parse_symbol(','))
            var_dec.append(self._parse_identifier())

        var_dec.append(self._parse_symbol(';'))

        return Token('varDec', var_dec)

    def _parse_statements(self):
        statements = []
        while self._check_token_val(self._STATEMENTS):
            if self._check_token_val('if'):
                statements.append(self._parse_if())
            elif self._check_token_val('while'):
                statements.append(self._parse_while())
            elif self._check_token_val('do'):
                statements.append(self._parse_do())
            elif self._check_token_val('let'):
                statements.append(self._parse_let())
            elif self._check_token_val('return'):
                statements.append(self._parse_return())
            else:
                raise ValueError(
                    f"Expected statement, got {self.tokens[self.idx]} at {self.idx}"
                )

        return Token('statements', statements)

    def _parse_let(self):
        let = [self._parse_keyword('let'), self._parse_identifier()]

        if self._check_token_val('['):
            let.append(self._parse_symbol('['))
            let.append(self._parse_expression())
            let.append(self._parse_symbol(']'))

        let.append(self._parse_symbol('='))
        let.append(self._parse_expression())
        let.append(self._parse_symbol(';'))

        return Token('letStatement', let)

    def _parse_if(self):
        if_statement = [
            self._parse_keyword('if'),
            self._parse_symbol('('),
            self._parse_expression(),
            self._parse_symbol(')'),
            self._parse_symbol('{'),
            self._parse_statements(),
            self._parse_symbol('}'),
        ]
        if self._check_token_val('else'):
            if_statement += [
                self._parse_keyword('else'),
                self._parse_symbol('{'),
                self._parse_statements(),
                self._parse_symbol('}'),
            ]
        return Token('ifStatement', if_statement)

    def _parse_while(self):
        return Token(
            'whileStatement',
            [
                self._parse_keyword('while'),
                self._parse_symbol('('),
                self._parse_expression(),
                self._parse_symbol(')'),
                self._parse_symbol('{'),
                self._parse_statements(),
                self._parse_symbol('}'),
            ],
        )

    def _parse_do(self):
        do = [self._parse_keyword('do')]
        do += self._parse_subroutine_call()
        do.append(self._parse_symbol(';'))
        return Token('doStatement', do,)

    def _parse_subroutine_call(self):
        subroutine_call = [self._parse_identifier()]
        if self._check_token_val('.'):
            subroutine_call.append(self._parse_symbol('.'))
            subroutine_call.append(self._parse_identifier())

        subroutine_call += [
            self._parse_symbol('('),
            self._parse_expression_list(),
            self._parse_symbol(')'),
        ]

        return subroutine_call

    def _parse_return(self):
        return_st = [self._parse_keyword('return')]
        if not self._check_token_val(';'):
            return_st.append(self._parse_expression())
        return_st.append(self._parse_symbol(';'))

        return Token('returnStatement', return_st)

    def _parse_expression(self):
        expression = [self._parse_term()]
        while self._check_token_val(self._OPS):
            expression.append(self._parse_op())
            expression.append(self._parse_term())
        return Token('expression', expression)

    def _parse_term(self):
        if self._check_token_type('integerConstant'):
            term = [self._parse_int_const()]
        elif self._check_token_type('stringConstant'):
            term = [self._parse_str_const()]
        elif self._check_token_val(self._KEYWORD_CONSTS):
            term = [self._parse_keyword()]
        elif self._check_token_val(self._UNARY_OPS):
            term = [self._parse_op(), self._parse_term()]
        elif self._check_token_val('('):
            term = [
                self._parse_symbol('('),
                self._parse_expression(),
                self._parse_symbol(')'),
            ]
        elif self._check_token_type('identifier'):
            term = [self._parse_identifier()]
            if self._check_token_val('['):  # array element
                term.append(self._parse_symbol('['))
                term.append(self._parse_expression())
                term.append(self._parse_symbol(']'))
            elif self._check_token_val('.'):  # another class subroutine
                term.append(self._parse_symbol('.'))
                term.append(self._parse_identifier())
                term.append(self._parse_symbol('('))
                term.append(self._parse_expression_list())
                term.append(self._parse_symbol(')'))
            if self._check_token_val('('):  # subroutine
                term.append(self._parse_symbol('('))
                term.append(self._parse_expression_list())
                term.append(self._parse_symbol(')'))
        else:
            raise ValueError(
                f"Failed to parse term: {self.tokens[self.idx]} at {self.idx}"
            )

        return Token('term', term)

    def _parse_expression_list(self):
        expression_list = []
        if not self._check_token_val(')'):
            expression_list = [self._parse_expression()]
        while self._check_token_val(','):
            expression_list.append(self._parse_symbol(','))
            expression_list.append(self._parse_expression())
        return Token('expressionList', expression_list)

    def parse(self, tokens, trim=False):
        logging.info(f"Parsing {len(tokens)} tokens")
        self.tokens = tokens
        parse_tree = self._parse_class()
        logging.info(f"Parsing done.")
        if trim:
            parse_tree = trim_children(parse_tree)
            logging.info("Removing empty entries")
        return parse_tree


class JackCompiler(object):
    def __init__(self):
        self.class_name = None
        self.class_vars = {}
        self.subroutine_vars = {}
        self.labels = []
        self.vm_code = []

    def _get_next_label(self):
        self.labels.append(f'L{len(self.labels)}')
        return self.labels[-1]

    @staticmethod
    def _build_xml_layer(parent_node, parent_token):
        if not isinstance(parent_token.val, list):
            if not parent_token.val:
                parent_node.text = '\n'
            else:
                parent_node.text = f" {parent_token.val} "
        else:
            for t in parent_token.val:
                child = ET.SubElement(parent_node, t.label)
                JackCompiler._build_xml_layer(child, t)

    def to_xml(self, parse_tree):
        root = ET.Element('class')
        self._build_xml_layer(root, parse_tree)
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

        return xmlstr[23:]  # Need to remove header for comparison files :(

    def to_vm(self, parse_tree):
        self._compile_class(parse_tree)
        return '\n'.join(self.vm_code)

    def _write_push(self, segment, idx):
        self.vm_code.append(f"push {segment} {idx}")
        return self.vm_code[-1]

    def _write_pop(self, segment, idx):
        self.vm_code.append(f"pop {segment} {idx}")
        return self.vm_code[-1]

    def _write_arithmetic(self, symbol):
        _ARITHMETIC = {
            '+': 'add',
            '-': 'sub',
            '*': 'call Math.multiply 2',
            '/': 'call Math.divide 2',
            '&': 'and',
            '|': 'or',
            '<': 'lt',
            '>': 'gt',
            '=': 'eq',
        }
        self.vm_code.append(_ARITHMETIC[symbol])
        return self.vm_code[-1]

    def _write_unary_op(self, symbol):
        _UNARY_OPS = {
            '-': 'neg',
            '~': 'not',
        }
        self.vm_code.append(_UNARY_OPS[symbol])
        return self.vm_code[-1]

    def _write_label(self, label):
        self.vm_code.append(f"label {label}")
        return self.vm_code[-1]

    def _write_goto(self, label):
        self.vm_code.append(f"goto {label}")
        return self.vm_code[-1]

    def _write_if(self, label):
        self.vm_code.append(f"if-goto {label}")
        return self.vm_code[-1]

    def _write_call(self, name, n_args):
        self.vm_code.append(f"call {name} {n_args}")
        return self.vm_code[-1]

    def _write_function(self, name, n_locals):
        self.vm_code.append(f"function {name} {n_locals}")
        return self.vm_code[-1]

    def _write_return(self):
        self.vm_code.append(f"return")
        return self.vm_code[-1]

    def _compile_class(self, token):
        self.class_name = token.val[1].val
        self.class_vars = {}
        i = 2
        while i < len(token.val):
            if token.val[i].label == 'classVarDec':
                self._compile_class_var_dec(token.val[i])
            elif token.val[i].label == 'subroutineDec':
                self._compile_subroutine_dec(token.val[i])
            i += 1

    def _compile_class_var_dec(self, token):
        kind = token.val[0].val
        type = token.val[1].val
        i = 2
        while i < len(token.val):
            self._add_class_var(token.val[i].val, type, kind)
            i += 1

    def _compile_subroutine_var_dec(self, token):
        kind = 'local'
        type = token.val[1].val
        i = 2
        while i < len(token.val):
            self._add_class_var(token.val[i].val, type, kind)
            i += 1

    def _compile_subroutine_dec(self, token):
        self.subroutine_vars = {}
        param_list = token.val[3].val
        if param_list:
            n_param = int(len(param_list) / 2)
            i = 0
            while i < len(param_list):
                self._add_subroutine_var(
                    param_list[i + 1].val, param_list[i].val, 'argument'
                )
                i += 2
        else:
            n_param = 0

        self._write_function(f"{self.class_name}.{token.val[2].val}", n_param)

        if token.val[0].val == 'method':
            self._write_push('argument', 0)
            self._write_pop('pointer', 0)
        elif token.val[0].val == 'constructor':
            self._write_push('constant', n_param)
            self._write_call('Memory.alloc', 1)
            self._write_pop('pointer', 0)
        subroutine_body = token.val[4].val
        if subroutine_body:
            i = 0
            while i < len(subroutine_body):
                if subroutine_body[i].label == 'varDec':
                    self._compile_subroutine_var_dec(subroutine_body[i])
                if subroutine_body[i].label == 'statements':
                    self._compile_statements(subroutine_body[i])
                i += 1

    def _compile_expression(self, token):
        term = token.val[0]
        self._compile_term(term)

        i = 1
        while i < len(token.val):
            self._compile_term(token.val[i + 1])
            self._write_arithmetic(token.val[i].val)
            i += 2

    def _compile_term(self, token):
        if len(token.val) == 1:
            if token.val[0].label == "integerConstant":
                self._compile_int_const(token.val[0])
            elif token.val[0].label == "stringConstant":
                self._compile_str_const(token.val[0])
            elif token.val[0].label == 'keywordConstant':
                self._compile_keyword_const(token.val[0])
            elif token.val[0].label == 'expression':
                self._compile_expression(token.val[0])
            elif token.val[0].label == "identifier":
                self._write_push(*self._fetch_var(token.val[0].val))
        else:
            if token.val[0].label == 'op':
                self._compile_term(token.val[1])
                self._write_unary_op(token.val[0].val)
            elif token.val[0].label == "identifier":
                if token.val[1].label == 'expression':
                    pass  # array
                else:
                    n_args = len(token.val[-1].val)
                    self._compile_expression_list(token.val[-1])
                    if token.val[1].label == 'identifier':
                        if self._fetch_var(token.val[0].val):
                            # Method call
                            self._write_push(self._fetch_var(token.val[0].val))
                            self._write_call(
                                f"{token.val[0].val}.{token.val[1].val}", n_args + 1
                            )
                        else:
                            # Function or constructor
                            self._write_call(
                                f"{token.val[0].val}.{token.val[1].val}", n_args
                            )
                    elif token.val[1].label == 'expressionList':
                        # Method call
                        self._write_push(self._fetch_var(token.val[0].val))
                        self._write_call(token.val[0].val, n_args + 1)

    def _compile_expression_list(self, token):
        for t in token.val:
            self._compile_expression(t)

    def _compile_int_const(self, token):
        self._write_push('constant', token.val)

    def _compile_str_const(self, token):
        self._write_push('constant', len(token.val))
        self._write_call('String.new', 1)
        for char in token.val:
            self._write_push('constant', ord(char))
            self._write_call('String.append', 2)

    def _compile_keyword_const(self, token):
        if token.val == 'null' or token.val == 'false':
            self._write_push('constant', 0)
        elif token.val == 'true':
            self._write_push('constant', 1)
            self._write_unary_op('-')
        elif token.val == 'this':
            self._write_push('pointer', 0)
    
    def _compile_statements(self, token):
        for statement in token.val:
            self._compile_statement(statement)

    def _compile_statement(self, token):
        if token.label == 'doStatement':
            self._compile_do(token)
        elif token.label == 'letStatement':
            self._compile_let(token)
        elif token.label == 'returnStatement':
            self._compile_return(token)
        elif token.label == 'ifStatement':
            self._compile_if(token)
        elif token.label == 'whileStatement':
            self._compile_while(token)

    def _compile_do(self, token):
        expression_list = token.val[-1]
        self._compile_expression_list(expression_list)
        n_args = len(expression_list.val)
        if len(token.val) == 3:
            self._write_push('this', 0)
            self._write_call(token.val[1].val, n_args)
        else:
            if self._fetch_var(token.val[1].val):
                self._write_push(self._fetch_var(token.val[1].val))
                self._write_call(f"{token.val[1].val}.{token.val[2].val}", n_args + 1)
            else:
                self._write_call(f"{token.val[1].val}.{token.val[2].val}", n_args)
        self._write_pop('temp', 0)

    def _compile_let(self, token):
        self._compile_expression(token.val[2])
        self._write_pop(*self._fetch_var(token.val[1].val))

    def _compile_return(self, token):
        if len(token.val) > 1:
            self._compile_expression(token.val[1])
        else:
            self._write_push('constant', 0)
        self._write_return()

    def _compile_if(self, token):
        self._compile_expression(token.val[1])
        self._write_unary_op('~')
        l1 = self._get_next_label()
        self._write_if(l1)
        self._compile_statements(token.val[2])
        l2 = self._get_next_label()
        self._write_goto(l2)
        self._write_label(l1)
        if len(token.val) > 3:
            self._compile_statements(token.val[4])
        self._write_label(l2)

    def _compile_while(self, token):
        l1 = self._get_next_label()
        self._write_label(l1)
        self._compile_expression(token.val[1])
        self._write_unary_op('~')
        l2 = self._get_next_label()
        self._write_if(l2)
        self._compile_statements(token.val[2])
        self._write_goto(l1)
        self._write_label(l2)

    def _fetch_var(self, name):
        if name in self.subroutine_vars:
            return (
                self.subroutine_vars[name]['kind'],
                self.subroutine_vars[name]['index'],
            )
        elif name in self.class_vars:
            return (
                self.class_vars[name]['kind'],
                self.class_vars[name]['index'],
            )
        else:
            logging.info(f"Failed to find var: '{name}'")

    def _var_count(self, scope, kind):
        if scope == 'class':
            return len([v for v in self.class_vars.values() if v['kind'] == kind])
        elif scope == 'subroutine':
            return len([v for v in self.subroutine_vars.values() if v['kind'] == kind])
        else:
            raise ValueError(f"Expected one of 'class', 'subroutine', got '{scope}'")

    def _add_class_var(self, name, type, kind):
        self.class_vars[name] = {
            'type': type,
            'kind': kind,
            'index': self._var_count('class', kind),
        }

    def _add_subroutine_var(self, name, type, kind):
        self.subroutine_vars[name] = {
            'type': type,
            'kind': kind,
            'index': self._var_count('subroutine', kind),
        }


def tokenize(filepath):
    tokenizer = JackTokenizer()
    for fp in fu.get_filepaths(filepath, '.jack'):
        tokens = tokenizer.tokenize_file(fp)
        for token in tokens:
            logging.info(f"  {token}")
        logging.info(f"Output {len(tokens)} tokens parsed from {fp}.")
        logging.info(f"{'-'*88}")


def analyze(filepath, trim=False):
    tokenizer = JackTokenizer()
    analyzer = JackAnalyzer()
    compiler = JackCompiler()
    for fp in fu.get_filepaths(filepath, '.jack'):
        tokens = tokenizer.tokenize_file(fp)
        parsed = analyzer.parse(tokens, trim)
        xml = compiler.to_xml(parsed)
        fu.to_file(xml, fp, '.xml')
        logging.info(f"{'-' * 88}")


def write_vm(filepath):
    tokenizer = JackTokenizer()
    analyzer = JackAnalyzer()
    compiler = JackCompiler()
    for fp in fu.get_filepaths(filepath, '.jack'):
        tokens = tokenizer.tokenize_file(fp)
        parsed = analyzer.parse(tokens, trim=True)
        vm_code = compiler.to_vm(parsed)
        fu.to_file(vm_code, fp, '.vm')
        logging.info(f"{'-' * 88}")


def main(filepath, mode):
    logging.info(f"Got args: {filepath}, {mode}")
    if mode == 'a':
        analyze(filepath, True)
    elif mode == 'c':
        write_vm(filepath=filepath)
    else:
        raise ValueError(f"Invalid 'mode' param. Got '{mode}', expected 'a' or 'c'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Jack code into VM code")
    parser.add_argument(
        "filepath",
        type=str,
        help="Input file or directory (files must have .jack extension)",
    )
    parser.add_argument(
        "mode",
        type=str,
        help="'a' to analyze, outputing .xml, 'c' to compile, outputing .vm",
    )
    args = parser.parse_args()
    main(args.filepath, args.mode)
