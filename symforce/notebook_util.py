"""
Helpers for interactive use in a Jupyter notebook with an IPython kernel.
"""
import IPython
import pygments

import sympy as sympy_py

sympy_py.init_printing()


def display(*args):
    """
    Display the given expressions in latex, or print if not an expression.

    Args:
        args (list):
    """
    try:
        IPython.display.display(sympy_py.S(*args))
    except (sympy_py.SympifyError, AttributeError, TypeError):
        IPython.display.display(*args)


def display_code(code, language):
    """
    Display code with syntax highlighting.

    Args:
        code (str): Source code
        language (str): {python, c++, anything supported by pygments}
    """
    lexer = pygments.lexers.get_lexer_by_name(language)
    formatter = pygments.formatters.HtmlFormatter(noclasses=True)
    html = pygments.highlight(code, lexer, formatter)

    IPython.display.display(IPython.display.HTML(html))


def display_code_file(path, language):
    """
    Display code from a file path with syntax highlighting.

    Args:
        path (str): Path to source file
        language (str): {python, c++, anything supported by pygments}
    """
    with open(path) as f:
        code = f.read()

    display_code(code, language)