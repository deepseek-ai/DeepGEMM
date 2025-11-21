import re
from pathlib import Path


def build_cpp_function_index(root_path):
    func_index = {}
    extensions = {'.cpp', '.cc', '.cxx', '.c', '.hpp', '.h'}

    pattern = re.compile(
        r'([\w:\s*<&>,\[\]\(\)]+?)'
        r'\s+'
        r'([a-zA-Z_][a-zA-Z0-9_:]*)'
        r'\s*\(',
    )

    for file_path in Path(root_path).rglob('*'):
        if file_path.suffix.lower() not in extensions:
            continue
        if not file_path.is_file():
            continue

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f'Failed to read file {file_path}: {e}')
            continue

        # Remove the compile directives and comments
        lines = content.split('\n')
        clean_lines = [line for line in lines if not line.strip().startswith(('#', '//'))]
        content = '\n'.join(clean_lines)

        for match in pattern.finditer(content):
            return_type_part = match.group(1).strip()
            full_func_name = match.group(2).strip()

            if not return_type_part or not re.match(r'^[a-zA-Z_]', return_type_part):
                continue

            first_token = return_type_part.split()[0]
            if first_token in {'return', 'if', 'else', 'for', 'while', 'switch', 'case', 'throw', 'catch', 'auto'}:
                continue

            # Extract base name
            if '::' in full_func_name:
                base_name = full_func_name.split('::')[-1]
            else:
                base_name = full_func_name

            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', base_name):
                continue

            # Find matching ')'
            paren_start = match.end() - 1
            paren_count = 0
            pos = paren_start
            while pos < len(content):
                ch = content[pos]
                if ch == '(':
                    paren_count += 1
                elif ch == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        break
                    elif paren_count < 0:
                        pos = -1
                        break
                pos += 1
            else:
                continue

            if pos == -1:
                continue

            # Check context before match: should be at statement boundary
            match_start = match.start()
            context_before = content[max(0, match_start - 50):match_start]
            if context_before and re.search(r'[a-zA-Z0-9_]$', context_before.rstrip()):
                continue

            # Check for definition or header declaration
            is_header = file_path.suffix.lower() in {'.h', '.hpp', '.cuh'}
            after_paren = content[pos+1:pos+500]
            has_brace = '{' in after_paren
            has_semicolon = ';' in after_paren.split('{')[0]

            if has_brace or (is_header and has_semicolon):
                sig_start = match.start(1)
                full_signature = content[sig_start:pos+1].strip()
                if base_name not in func_index:
                    func_index[base_name] = full_signature

    return func_index


class BracketTracker:
    """
    Tracks nesting levels of various brackets in C++ code:
      - () → paren
      - [] → bracket
      - {} → brace
      - <> → angle (treated as template brackets only at top level)
    Provides is_top_level() to check if currently outside all brackets.
    """
    def __init__(self):
        self.paren = 0      # ()
        self.bracket = 0    # []
        self.brace = 0      # {}
        self.angle = 0      # <>

    def update(self, char: str):
        """
        Update internal counters based on the given character.
        """
        if char == '(':
            self.paren += 1
        elif char == ')':
            self.paren -= 1
        elif char == '[':
            self.bracket += 1
        elif char == ']':
            self.bracket -= 1
        elif char == '{':
            self.brace += 1
        elif char == '}':
            self.brace -= 1
        # Angle brackets < > are only treated as template delimiters
        # when not inside (), [], or {}
        elif char == '<' and self._in_top_level_of_other_brackets():
            self.angle += 1
        elif char == '>' and self.angle > 0 and self._in_top_level_of_other_brackets():
            self.angle -= 1

    def _in_top_level_of_other_brackets(self):
        """
        Check if not inside parentheses, square brackets, or braces (for correct template bracket recognition).
        """
        return self.paren == 0 and self.bracket == 0 and self.brace == 0

    def is_top_level(self):
        """
        Check if completely at top level (all bracket counters are zero).
        """
        return (self.paren == 0 and
                self.bracket == 0 and
                self.brace == 0 and
                self.angle == 0)


def extract_m_def_statements(root_path):
    """
    Scan all c files under root_path and extract all m.def(...) statements.
    """
    results = []
    extensions = {'.hpp', '.cpp', '.h', '.cc'}

    # Regex: match m.def( ... ), supports multi-line
    pattern = re.compile(r'm\.def\s*\(')

    for file_path in Path(root_path).rglob('*'):
        if file_path.suffix.lower() not in extensions:
            continue
        if not file_path.is_file():
            continue

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f'Failed to read file {file_path}: {e}')
            continue

        m_def_list = []
        lines = content.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'm.def(' in line:
                # Found a potential starting line
                start_i = i
                # Check if it's a comment
                stripped = line.lstrip()
                if stripped.startswith('//') or stripped.startswith('/*'):
                    i += 1
                    continue

                # Try to match the complete m.def(...) call
                paren_count = 0
                j = i
                found_start = False
                while j < len(lines):
                    current_line = lines[j]
                    for k, char in enumerate(current_line):
                        if char == '(':
                            if not found_start and re.search(r'm\.def\s*\(', current_line[:k+1]):
                                found_start = True
                            if found_start:
                                paren_count += 1
                        elif char == ')':
                            if found_start:
                                paren_count -= 1
                                if paren_count == 0:
                                    # Found complete statement
                                    full_stmt = ''.join(lines[i:j+1]).rstrip()
                                    m_def_list.append(full_stmt)
                                    i = j
                                    break
                    if paren_count <= 0 and found_start:
                        break
                    j += 1
                else:
                    pass
            i += 1

        if m_def_list:
            results.append({
                'file': str(file_path),
                'm_def_statements': m_def_list
            })

    return results


def parse_m_def_statement(m_def_str):
    result = {
        'python_function_name': None,
        'num_args': 0,
        'default_args': {},
        'is_lambda': False,
    }

    # Extract top-level arguments
    start = m_def_str.find('m.def(')
    if start == -1:
        raise ValueError(f'[{m_def_str}] Could not find m.def start position')

    paren_count = 0
    content_start = start + len('m.def(')
    content_end = -1
    for i in range(content_start, len(m_def_str)):
        ch = m_def_str[i]
        if ch == '(':
            paren_count += 1
        elif ch == ')':
            if paren_count == 0:
                content_end = i
                break
            else:
                paren_count -= 1
    if content_end == -1:
        raise ValueError(f'[{m_def_str}] m.def parentheses not closed')

    args_content = m_def_str[content_start:content_end]

    # Split arguments using BracketTracker
    args_list = []
    current = []
    tracker = BracketTracker()

    for ch in args_content:
        if ch in '()[]{}<>':
            tracker.update(ch)
        if ch == ',' and tracker.is_top_level():
            args_list.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        args_list.append(''.join(current).strip())

    if len(args_list) < 2:
        raise ValueError(f'[{m_def_str}] m.def has insufficient arguments')

    # Extract Python function name
    first = args_list[0].strip()
    str_match = re.match(r'^"([^"\\]*(?:\\.[^"\\]*)*)"', first)
    if str_match:
        result['python_function_name'] = str_match.group(1)
    else:
        raise ValueError(f'[{m_def_str}] m.def first argument should be a string literal')

    cpp_func_part = args_list[1].strip()
    if cpp_func_part.startswith('&'):
        cpp_func_part = cpp_func_part[1:].strip()

    if cpp_func_part.startswith('['):
        result['is_lambda'] = True
        result['cpp_function_name'] = None
    else:
        if '::' in cpp_func_part:
            cpp_func_name = cpp_func_part.split('::')[-1]
        else:
            cpp_func_name = cpp_func_part

        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)', cpp_func_name)
        if match:
            result['cpp_function_name'] = match.group(1)
        else:
            result['cpp_function_name'] = cpp_func_name

    # Parse py::arg arguments
    py_args = args_list[2:]
    result['num_args'] = len(py_args)

    for idx, arg_expr in enumerate(py_args):
        expr = arg_expr.strip()
        # Find top-level '='
        eq_pos = -1
        p_depth = b_depth = br_depth = angle_depth = 0
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch == '(':
                p_depth += 1
            elif ch == ')':
                p_depth -= 1
            elif ch == '[':
                b_depth += 1
            elif ch == ']':
                b_depth -= 1
            elif ch == '{':
                br_depth += 1
            elif ch == '}':
                br_depth -= 1
            elif ch == '<' and p_depth == 0 and b_depth == 0 and br_depth == 0:
                angle_depth += 1
            elif ch == '>' and angle_depth > 0 and p_depth == 0 and b_depth == 0 and br_depth == 0:
                angle_depth -= 1
            elif ch == '=' and all(d == 0 for d in [p_depth, b_depth, br_depth, angle_depth]):
                eq_pos = i
                break
            i += 1

        if eq_pos != -1:
            default_val = expr[eq_pos + 1:].strip()
            if not default_val:
                raise ValueError(f'[{expr}] Default value is empty (arg {idx})')
            result['default_args'][idx] = default_val

    return result


def extract_cpp_signature_from_content(cpp_func_name, content):
    """
    Search for the C++ function signature of cpp_func_name in the given file content.
    """
    if not cpp_func_name:
        return None

    # Build regex: match function starting with cpp_func_name (after word boundary)
    # Note: function name may be preceded by return type (with templates, namespaces, etc.), followed by '('
    pattern = re.compile(
        r'^\s*'                                        # leading whitespace
        r'([\w:\s*<&>,\[\]\(\)]+?)'                    # return type (non-greedy, allows templates, pointers, etc.)
        r'\s+'                                         # at least one space
        r'\b' + re.escape(cpp_func_name) + r'\b'       # function name (word boundary)
                                           r'\s*\(',   # optional whitespace + start of param list
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        # Find '(' position after function name
        paren_start = match.end() - 1
        if content[paren_start] != '(':
            paren_start = content.find('(', match.end(0) - 1)
            if paren_start == -1:
                continue

        # From '(', match to corresponding ')'
        paren_count = 0
        pos = paren_start
        while pos < len(content):
            ch = content[pos]
            if ch == '(':
                paren_count += 1
            elif ch == ')':
                paren_count -= 1
                if paren_count == 0:
                    start_sig = match.start(1)
                    full_signature = content[start_sig:pos+1].strip()
                    return full_signature
            pos += 1

    return None


def parse_mdef_and_attach_cpp_signatures(item, func_index):
    """
    Enhance item by parsing m.def and extracting C++ function signature from global index
    """
    statements_with_parsed_signatures = []

    for stmt in item['m_def_statements']:
        parsed = parse_m_def_statement(stmt,)
        cpp_func_name = parsed.get('cpp_function_name')

        cpp_sig = None
        if cpp_func_name and cpp_func_name in func_index:
            cpp_sig = func_index[cpp_func_name]
        else:
            if not parsed['is_lambda']:
                print(f'Warning: C++ function "{cpp_func_name}" not found in any .cpp file')

        parsed['cpp_signature'] = cpp_sig
        statements_with_parsed_signatures.append({
            'raw': stmt,
            'parsed': parsed
        })

    return {
        'm_def_statements': statements_with_parsed_signatures
    }


def parse_cpp_signature(cpp_sig):
    """
    Parse a C++ function signature and extract return type, parameter types, and names.
    """
    if not cpp_sig or not cpp_sig.strip():
        return None

    # Find function name: last identifier before '('
    paren_pos = cpp_sig.find('(')
    if paren_pos == -1:
        return None

    before_paren = cpp_sig[:paren_pos].strip()
    if not before_paren:
        return None

    # Function name is the last word in before_paren (may include templates like func<int>)
    tokens = before_paren.split()
    if len(tokens) < 2:
        return None

    # Heuristic: function name is usually the last token (may include <>)
    func_name_part = tokens[-1]
    return_type = ' '.join(tokens[:-1]).strip()

    # Now extract parameter list content
    param_list_str = cpp_sig[paren_pos+1:cpp_sig.rfind(')')].strip()
    parameters = []

    if param_list_str and param_list_str != 'void':  # 'void' means no parameters
        # Split parameters (handle commas not inside templates/brackets)
        param_decls = split_cpp_parameters(param_list_str)
        for decl in param_decls:
            decl = decl.strip()
            if not decl:
                continue
            # Try to split type and name from right to left
            param_info = parse_parameter_declaration(decl)
            if param_info:
                parameters.append(param_info)

    return {
        'return_type': return_type,
        'parameters': parameters,
        'num_parameters': len(parameters)
    }


def split_cpp_parameters(param_str: str):
    """
    Split a C++ parameter list string by top-level commas,
    e.g., 'int a, std::vector<float> b' → ['int a', 'std::vector<float> b']
    """
    if not param_str.strip() or param_str == 'void':
        return []
    params = []
    current = []
    tracker = BracketTracker()

    for ch in param_str:
        if ch in '()[]{}<>':
            tracker.update(ch)
        if ch == ',' and tracker.is_top_level():
            param = ''.join(current).strip()
            if param:  # Only add non-empty parameters
                params.append(param)
            current = []
        else:
            current.append(ch)

    if current:
        final_param = ''.join(current).strip()
        if final_param:  # Only add non-empty parameters
            params.append(final_param)
    return params


def parse_parameter_declaration(decl: str):
    """
    Parse a single parameter declaration, e.g., 'const std::string& name' → {'type': 'const std::string&', 'name': 'name'}
    Improved version that better handles template types.
    """
    decl = decl.strip()
    if not decl:
        return None

    # Remove possible default value (starting from top-level '=')
    tracker = BracketTracker()
    eq_pos = -1
    for i, ch in enumerate(decl):
        if ch in '()[]{}<>':
            tracker.update(ch)
        elif ch == '=' and tracker.is_top_level():
            eq_pos = i
            break

    if eq_pos != -1:
        decl = decl[:eq_pos].strip()

    # Now decl is 'type name' or just 'type'
    # Instead of simple splitting, we'll use a more robust approach
    # to find the parameter name

    # First, let's handle the case where there's no explicit parameter name
    # (this sometimes happens in function declarations)
    if not re.search(r'[a-zA-Z_][a-zA-Z0-9_]*$', decl):
        # No parameter name found, just return the type
        return {
            'type': decl,
            'name': None
        }

    # Use bracket tracking to find where the type ends and name begins
    tracker = BracketTracker()
    name_start = -1

    # Scan from the end to find the start of the parameter name
    # We look for the first identifier that's outside all brackets
    i = len(decl) - 1
    while i >= 0:
        ch = decl[i]

        if ch in '()[]{}<>':
            tracker.update(ch)

        # If we're at top level and find an identifier character
        if tracker.is_top_level() and re.match(r'[a-zA-Z0-9_]', ch):
            # Track back to find the start of this identifier
            name_start = i
            while name_start > 0 and re.match(r'[a-zA-Z0-9_]', decl[name_start - 1]):
                name_start -= 1

            # Check if this might be part of a type keyword (like 'int', 'bool', etc.)
            potential_name = decl[name_start:i+1]
            type_keywords = {'int', 'long', 'short', 'char', 'bool', 'float', 'double',
                             'void', 'auto', 'const', 'static', 'volatile', 'mutable',
                             'unsigned', 'signed'}

            # If it's not a type keyword and looks like a parameter name, use it
            if (potential_name not in type_keywords and
                    re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', potential_name)):
                break

        i -= 1

    if name_start != -1 and i >= 0:
        param_name = decl[name_start:i+1]
        param_type = decl[:name_start].strip()

        # Clean up the type - remove trailing &, * and whitespace
        param_type = param_type.rstrip('&* \t')

        return {
            'type': param_type,
            'name': param_name
        }

    # Fallback: if we can't find a clear parameter name, just return the type
    return {
        'type': decl,
        'name': None
    }


def extract_cpp_signature_details(item):
    """
    For each m.def entry in item, parse cpp_signature to extract return type and parameter details.
    """
    statements_with_parsed_signatures = []
    for stmt_info in item['m_def_statements']:
        parsed = stmt_info['parsed']
        cpp_sig = parsed.get('cpp_signature')

        cpp_params_info = None
        if cpp_sig:
            try:
                cpp_params_info = parse_cpp_signature(cpp_sig)
            except Exception as e:
                print(f'Failed to parse C++ signature: {e}')

        parsed['cpp_parsed_signature'] = cpp_params_info
        statements_with_parsed_signatures.append({
            'raw': stmt_info['raw'],
            'parsed': parsed
        })

    return {
        'm_def_statements': statements_with_parsed_signatures
    }


def cpp_type_to_python_type(cpp_type: str) -> str:
    if not cpp_type:
        return 'Any'

    original = cpp_type.strip()
    if not original:
        return 'Any'

    # Remove C++ specifiers that don't affect Python type
    cleaned = re.sub(r'\b(static|inline|constexpr|thread_local|extern|mutable|const|volatile|endif)\b', '', original)
    cleaned = cleaned.replace('&', '').replace('*', '').strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Handle void
    if cleaned == 'void':
        return 'None'

    # Handle template types — ORDER MATTERS! Must come before internal type checks.

    # std::pair<T1, T2>
    if cleaned.startswith('std::pair<'):
        inner = cleaned[10:-1].strip()  # len('std::pair<') == 10
        args = split_template_args(inner)
        if len(args) == 2:
            t1 = cpp_type_to_python_type(args[0])
            t2 = cpp_type_to_python_type(args[1])
            return f'tuple[{t1}, {t2}]'
        else:
            print(f'Warning: std::pair with unexpected number of args: {cleaned}')
            return 'Any'

    # std::tuple<T1, T2, ...>
    if cleaned.startswith('std::tuple<'):
        inner = cleaned[11:-1].strip()  # len('std::tuple<') == 11
        args = split_template_args(inner)
        py_types = [cpp_type_to_python_type(arg) for arg in args]
        return f"tuple[{', '.join(py_types)}]"

    # std::vector<T>
    if cleaned.startswith('std::vector<'):
        inner = cleaned[12:-1].strip()  # len('std::vector<') == 12
        args = split_template_args(inner)
        if len(args) == 1:
            inner_py = cpp_type_to_python_type(args[0])
            return f'list[{inner_py}]'
        else:
            print(f'Warning: std::vector with unexpected args: {cleaned}')
            return 'Any'

    # std::optional<T>
    if cleaned.startswith('std::optional<'):
        inner = cleaned[14:-1].strip()  # len('std::optional<') == 14
        args = split_template_args(inner)
        if len(args) == 1:
            inner_py = cpp_type_to_python_type(args[0])
            return f'Optional[{inner_py}]'
        else:
            print(f'Warning: std::optional with unexpected args: {cleaned}')
            return 'Any'

    # std::string
    if re.search(r'\bstd::string\b', original):
        return 'str'

    # C-style strings: char*, const char*, char[], etc.
    if re.search(r'\b(?:const\s+)?char\s*[\*\[]', original):
        return 'str'

    # Boolean
    if re.search(r'\bbool\b', cleaned):
        return 'bool'

    # Integer types (including fixed-width and common aliases)
    if re.search(r'\b(int|long|short|size_t|ssize_t|ptrdiff_t|'
                 r'int8_t|int16_t|int32_t|int64_t|'
                 r'uint8_t|uint16_t|uint32_t|uint64_t)\b', cleaned):
        return 'int'

    # Floating-point
    if re.search(r'\b(float|double|long\s+double)\b', cleaned):
        return 'float'

    # torch::Tensor
    if re.search(r'\btorch::Tensor\b', original):
        return 'torch.Tensor'

    # Unrecognized type
    print(f'Warning: Unrecognized C++ type: {original}')
    return 'Any'


def split_template_args(template_args: str):
    """
    Split template arguments, e.g., 'int, std::vector<float>' → ['int', 'std::vector<float>']
    """
    if not template_args.strip():
        return []
    args = []
    current = []
    tracker = BracketTracker()

    for ch in template_args:
        if ch in '()[]{}<>':
            tracker.update(ch)
        if ch == ',' and tracker.is_top_level():
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        args.append(''.join(current).strip())
    return args


def cpp_default_to_python_default(cpp_default: str):
    """
    Convert C++ default value string to valid Python expression string.
    """
    if not cpp_default:
        return 'None'

    s = cpp_default.strip()

    # Handle string literals: 'bf16' → 'bf16'
    # Match: starts and ends with unescaped double quotes
    string_match = re.match(r'^"([^"\\]*(?:\\.[^"\\]*)*)"$', s)
    if string_match:
        return s

    # Handle boolean literals
    if s == 'false':
        return 'False'
    if s == 'true':
        return 'True'

    # Handle null-like values: nullptr, nullopt, NULL, etc.
    if s in ('nullptr', 'NULL') or 'nullopt' in s:
        return 'None'

    # Handle std::tuple<int, int>({128, 128}) → (128, 128)
    tuple_match = re.match(r'std::tuple\s*<[^>]*>\s*\(\s*({.*?})\s*\)', s)
    if tuple_match:
        inner = tuple_match.group(1)  # {128, 128}
        inner_py = inner.replace('{', '(').replace('}', ')')
        return inner_py

    # Handle std::make_tuple(1, 2, 3) → (1, 2, 3)
    make_tuple_match = re.match(r'std::make_tuple\s*\(\s*(.*?)\s*\)', s)
    if make_tuple_match:
        inner = make_tuple_match.group(1)
        # Ensure it's a valid tuple even with one element: add comma if needed?
        # But in C++ default args, it's usually multi-element, so we assume valid.
        return f'({inner})'

    # Handle std::vector<int>({1,2,3}) → [1, 2, 3]
    vector_match = re.match(r'std::vector\s*<[^>]*>\s*\(\s*({.*?})\s*\)', s)
    if vector_match:
        inner = vector_match.group(1)
        inner_py = inner.replace('{', '[').replace('}', ']')
        return inner_py

    # Handle numeric literals: integers and floats
    if re.match(r'^[+-]?\d+$', s):  # integer
        return s
    if re.match(r'^[+-]?\d*\.\d+([eE][+-]?\d+)?$', s):  # float
        return s

    # Fallback: unrecognized → warn and return None
    print(f'Warning: Unrecognized default value: {s}')
    return 'None'


def generate_pyi_function(item_entry):
    parsed = item_entry['parsed']
    py_name = parsed['python_function_name']

    if parsed.get('is_lambda'):
        return f'def {py_name}(*args, **kwargs) -> Any: ...'

    sig_info = parsed.get('cpp_parsed_signature')
    default_args = parsed.get('default_args', {})

    if not sig_info:
        return f'def {py_name}(*args, **kwargs) -> Any: ...'

    return_type = cpp_type_to_python_type(sig_info['return_type'])
    params = sig_info['parameters']
    num_params = len(params)

    # Build parameter list
    param_lines = []
    for i in range(num_params):
        param_info = params[i] if i < len(params) else {'type': 'Any', 'name': f'arg{i}'}
        param_type = cpp_type_to_python_type(param_info['type'])
        param_name = param_info['name'] or f'arg{i}'

        # Replace invalid Python identifiers (e.g., keywords)
        if param_name in {'def', 'class', 'from', 'import', 'None', 'True', 'False'}:
            param_name = f'{param_name}_'

        # Check for default value
        if i in default_args:
            cpp_default = default_args[i]
            py_default = cpp_default_to_python_default(cpp_default)
            param_str = f'    {param_name}: {param_type} = {py_default}'
        else:
            param_str = f'    {param_name}: {param_type}'

        param_lines.append(param_str)

    if param_lines:
        params_block = ',\n'.join(param_lines)
        func_def = f'def {py_name}(\n{params_block}\n) -> {return_type}: ...'
    else:
        func_def = f'def {py_name}() -> {return_type}: ...'

    return func_def


def generate_pyi_file_content(enhanced_results, module_name: str = 'my_module'):
    function_decls = []
    has_optional = False
    has_torch = False
    has_numpy = False

    for item in enhanced_results:
        for stmt in item['m_def_statements']:
            try:
                decl = generate_pyi_function(stmt)
                function_decls.append(decl)

                if 'Optional[' in decl:
                    has_optional = True
                if 'torch.Tensor' in decl:
                    has_torch = True
                if 'numpy.ndarray' in decl or 'py::array' in str(stmt):
                    has_numpy = True
            except Exception as e:
                func_name = stmt['parsed'].get('python_function_name', 'unknown')
                function_decls.append(f'# ERROR: failed to generate stub for {func_name}: {e}')

    imports = ['from typing import Any']
    if has_optional:
        imports[0] += ', Optional'

    if has_torch:
        imports.append('import torch')
    if has_numpy:
        imports.append('import numpy')

    lines = [f'# Stubs for module: {module_name}', '']
    lines.extend(imports)
    lines.append('')
    lines.append('')

    for decl in function_decls:
        lines.append(decl)
        lines.append('')
        lines.append('')

    return '\n'.join(lines)


def generate_pyi_file(name, root, output_dir='.'):
    func_index = build_cpp_function_index(root)
    results = extract_m_def_statements(root)

    cpp_results = []
    for item in results:
        enhanced_item = parse_mdef_and_attach_cpp_signatures(item, func_index)
        cpp_item = extract_cpp_signature_details(enhanced_item)
        cpp_results.append(cpp_item)

    pyi_content = generate_pyi_file_content(cpp_results, module_name=name)

    output_path = Path(output_dir) / f'{name}.pyi'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pyi_content)

    print(f'.pyi file generated: {output_path}')
