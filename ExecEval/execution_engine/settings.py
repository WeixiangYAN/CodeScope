import re
from typing import Callable


def init_sanitize_c_cpp() -> Callable[[str], str]:
    _cpp_variant_lib_pattern = re.compile(r"variant\s*<.+>")

    def sanitize(source_code: str) -> str:
        stripped = lambda s: "".join(i for i in s if ord(i) < 128)
        source_code = stripped(source_code)
        # _getchar_nolock, _getwchar_nolock, __int64 issue
        source_code = (
            source_code.replace("_getchar_nolock", "getchar_unlocked")
            .replace("_getwchar_nolock", "getwchar_unlocked")
            .replace("_putc_nolock", "putc_unlocked")
            .replace("_putwc_nolock", "putwc_unlocked")
            .replace("_getc_nolock", "getc_unlocked")
            .replace("_getwc_nolock", "getwc_unlocked")
            .replace("_putchar_nolock", "putchar_unlocked")
            .replace("_putwchar_nolock", "putwchar_unlocked")
            .replace("__int64", "long long")
            .replace("__popcnt", "__builtin_popcount")
            .replace("__popcnt64", "__builtin_popcountll")
        )
        # gets to fgets
        source_code = re.sub(
            r"[^f]gets\((.+)\)", r"fgets(\1, sizeof(\1), stdin)", source_code
        )
        # <bits/stdC++.h>

        # variant library visit issue c++17
        # find if variant is used
        #### Major change by us
        variant_used = bool(_cpp_variant_lib_pattern.search(source_code) != None)
        if not variant_used:
            source_code = source_code.replace("visit", "__visit")

        _lines = source_code.split("\n")
        _sanitized_lines = list()
        for _line in _lines:
            # pragma issue
            if _line.startswith("#pragma GCC"):
                _sanitized_lines.append(" // " + _line)
            elif "<windows.h>" in _line:
                continue
            elif "<intrin.h>" in _line:
                continue
            elif _line.strip().startswith("#include"):
                _sanitized_lines.append(_line.replace("\\", "/").lower())
            elif "#define _GLIBCXX_DEBUG" in _line:
                continue
            else:
                _sanitized_lines.append(_line)

        return "\n".join(_sanitized_lines)

    return sanitize


def sanitize_kotlin(source_code: str) -> str:
    source_code = source_code.replace(".min()", ".minOrNull()")

    return source_code


sanitize_c_cpp = init_sanitize_c_cpp()

generic_c_cpp_compile = lambda s, cmd, flags: (
    f"{cmd} {flags} {s.name} -o {s.stem}.out",
    s.parent / f"{s.stem}.out",
)

generic_cs_compile = lambda s, cmd, flags: (
    f"{cmd} {flags} {s.name}",
    s.parent / f"{s.stem}.exe",
)

generic_java_compile = lambda s, cmd, flags: (
    f"{cmd} {flags} -d {s.parent} {s.name} ",
    s.parent / s.stem,
)

generic_kt_compile = lambda s, cmd, flags: (
    f"{cmd} {flags} -d {s.parent}/test.jar {str(s)}",
    s.parent / "test.jar",
)

generic_interpreted_compile = lambda s, cmd, flags: (f"{cmd} {flags} {s.name}", s)

generic_rust_go_compile = lambda s, cmd, flags: (
    f"{cmd} {flags} {s.name}",
    s.parent / f"{s.stem}",
)


class JavaClassNotFoundError(Exception):
    msg: str
    def __init__(self, msg, *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
        self.msg = msg

def init_java_file_name_suffix():
    _java_class_pattern = re.compile(r"public.* class (\w+)(.|\n|\r\n)*{", re.MULTILINE)

    def _java_file_name_suffix(source_code: str) -> str:
        result = _java_class_pattern.search(source_code)
        if result is None:
            return JavaClassNotFoundError("Failed to parse class name from:\n" + source_code)

        return result.group(1).strip() + ".java"

    return _java_file_name_suffix


java_file_name_suffix = init_java_file_name_suffix()

generic_binary_execute = lambda x, _, __: str(x)

generic_interpreted_execute = lambda x, cmd, flags: f"{cmd} {flags} {x}"

generic_java_execute = (
    lambda x, cmd, flags: f"{cmd} {flags} -cp {x.parent} {x.stem}"
)

generic_kotlin_execute = (
    lambda x, cmd, flags: f"{cmd} {flags} {str(x)}"
)