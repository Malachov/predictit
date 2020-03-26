
_JUPYTER = 0
_GUI = 0

try:
    __IPYTHON__
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('%load_ext autoreload')
    ipython.magic('%autoreload 2')
    _JUPYTER = 1

except Exception:
    pass


def traceback_warning(message=''):
    import pygments
    import textwrap
    import warnings
    import traceback

    separated_traceback = pygments.highlight(traceback.format_exc(), pygments.lexers.PythonTracebackLexer(), pygments.formatters.TerminalFormatter())
    separated_traceback = textwrap.indent(text=f"{message}\n====================\n\n{separated_traceback}\n====================\n", prefix='    ', predicate=lambda line: True)

    warnings.warn(f"\n\n{separated_traceback}")
