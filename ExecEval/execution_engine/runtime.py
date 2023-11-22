from pathlib import Path
from typing import Callable

import settings
from config import LanguageConfig


class Runtime:
    language: str
    compile_cmd: str = ""
    execute_cmd: str = ""
    compile_flags: str = ""
    execute_flags: str = ""
    file_name: Callable[[str], str] | str
    sanitize: Callable[[str], str] | None
    _compile: Callable[[Path, str, str], tuple[str | None, Path]] | None
    _execute: Callable[[Path, str, str], str]
    timelimit_factor: int = 1
    extend_mem_for_vm: bool = False
    extend_mem_flag_name: str = ""

    def __init__(self, cfg: LanguageConfig):
        self.language = cfg.language
        self.compile_cmd = cfg.compile_cmd
        self.compile_flags = cfg.compile_flags
        self.execute_cmd = cfg.execute_cmd
        self.execute_flags = cfg.execute_flags
        self.timelimit_factor = cfg.timelimit_factor
        self.file_name = getattr(
            settings, cfg.file_name_fn_or_str_name, cfg.file_name_fn_or_str_name
        )
        self.sanitize = getattr(settings, cfg.sanitize_fn_name, None)
        self._compile = getattr(settings, cfg.compile_fn_name, None)
        self._execute = getattr(settings, cfg.execute_fn_name, lambda _, __, ___: "")
        self.extend_mem_for_vm = cfg.extend_mem_for_vm
        self.extend_mem_flag_name = cfg.extend_mem_flag_name

    @property
    def is_compiled_language(self):
        return self._compile is not None

    @property
    def has_sanitizer(self):
        return self.sanitize is not None

    def get_info(self):
        return dict(
            runtime_name=self.language,
            compile_cmd=self.compile_cmd,
            compile_flags=self.compile_flags,
            execute_cmd=self.execute_cmd,
            execute_flags=self.execute_flags,
            timelimit_factor=self.timelimit_factor,
            is_compiled=self.is_compiled_language,
            has_sanitizer=self.has_sanitizer,
        )

    def get_file_path(self, source_code: str) -> Path | settings.JavaClassNotFoundError:
        if isinstance(self.file_name, str):
            return Path(self.file_name)

        file_name = self.file_name(source_code)
        
        if isinstance(file_name, settings.JavaClassNotFoundError):
            return file_name

        return Path(file_name)

    def compile(
        self,
        source_code_path: Path,
        cmd: str | None = None,
        flags: str | None = None,
    ):
        if self._compile is None:
            return [None, source_code_path]

        return self._compile(
            source_code_path,
            self.compile_cmd if cmd is None else cmd,
            ("" if self.compile_flags is None else self.compile_flags) + " " + ("" if flags is None else flags),
        )

    def execute(
        self, executable: Path, cmd: str | None = None, flags: str | None = None
    ):
        return self._execute(
            executable,
            self.execute_cmd if cmd is None else cmd,
            ("" if self.execute_flags is None else self.execute_flags) + " " + ("" if flags is None else flags),
        )
