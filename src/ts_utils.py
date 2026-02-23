
import tree_sitter
import tree_sitter as ts

import tree_sitter_java as ts_java
import tree_sitter_javascript as ts_js
import tree_sitter_c_sharp as ts_cs
import tree_sitter_python as ts_py
import tree_sitter_cpp as ts_cpp
import tree_sitter_c as ts_c
import tree_sitter_rust as ts_rust
import tree_sitter_php as ts_php

java_lang = ts.Language(ts_java.language())
java_parser = ts.Parser(java_lang)
js_lang = ts.Language(ts_js.language())
js_parser = ts.Parser(js_lang)
cs_lang = ts.Language(ts_cs.language())
cs_parser = ts.Parser(cs_lang)
py_lang = ts.Language(ts_py.language())
py_parser = ts.Parser(py_lang)
cpp_lang = ts.Language(ts_cpp.language())
cpp_parser = ts.Parser(cpp_lang)
c_lang = ts.Language(ts_c.language())
c_parser = ts.Parser(c_lang)
# rust_lang = ts.Language(ts_rust.language())
# rust_parser = ts.Parser(rust_lang)
# php_lang = ts.Language(ts_php.language_php())
# php_parser = ts.Parser(php_lang)


lang2function_node_name = {
    'javascript': 'function_declaration',
    'python': 'function_definition',
    'java': 'method_declaration',
    'cpp': 'function_definition',
    'c': 'function_definition',
}


lang_parser_map = {
    "java": java_parser,
    "javascript": js_parser,
    "csharp": cs_parser,
    "python": py_parser,
    "cpp": cpp_parser,
    "c": c_parser,
    # "rust": rust_parser,
    # "php": php_parser
}


def find_first_recursively_opt(node, type_name, depth=0):
    if node.type == type_name:
        return node
    if depth > 20:
        return None
    for child in node.named_children:
        result = find_first_recursively_opt(child, type_name, depth + 1)
        if result is not None:
            return result
    return None

def find_all_recursively(node, type_name, depth=0):
    result = []
    if node.type == type_name:
        result.append(node)
    if depth > 20:
        return result
    for child in node.named_children:
        result.extend(find_all_recursively(child, type_name, depth + 1))
    return result

def get_first_opt(node, type_name):
    for child in node.named_children:
        if child.type == type_name:
            return child
    return None

def get_all(node, type_name):
    result = []
    for child in node.named_children:
        if child.type == type_name:
            result.append(child)
    return result