from dataclasses import dataclass
from pathlib import Path

from helper import has_nested_dataclass
from resource_limit import ResourceLimits
from yaml import safe_load


@dataclass
class CodeStoreConfig:
    source_code_dir: Path

    def __post_init__(self):
        """Ensure its a Path instance"""
        self.source_code_dir = Path(self.source_code_dir)


@dataclass
class LanguageConfig:
    language: str = ""
    compile_cmd: str = ""
    compile_flags: str = ""
    execute_cmd: str = ""
    execute_flags: str = ""
    sanitize_fn_name: str = ""
    compile_fn_name: str = ""
    execute_fn_name: str = ""
    file_name_fn_or_str_name: str = ""
    timelimit_factor: int = 1
    extend_mem_for_vm: bool = False
    extend_mem_flag_name: str = ""


@has_nested_dataclass
class Config:
    supported_languages: dict[str, LanguageConfig]
    code_store: CodeStoreConfig
    run_uid: int
    run_gid: int

    def __init__(
        self,
        code_store: dict[str, str],
        supported_languages: dict[str, dict[str, str]],
        *args,
        **kwargs
    ):
        tmp = supported_languages.copy()
        self.supported_languages = dict()
        for lang, cfg in tmp.items():
            self.supported_languages[lang] = LanguageConfig(language=lang, **cfg)

        self.code_store = CodeStoreConfig(**code_store.__dict__)

        super().__init__(*args, **kwargs)


def load_config(config_file: Path) -> Config:
    with config_file.open("r") as f:
        cfg = Config(**safe_load(f))

    return cfg


def load_limits_by_lang(limits_by_lang_file: Path) -> dict[str, ResourceLimits]:
    limits_by_lang = dict()
    with open(limits_by_lang_file) as lblp:
        for lang, limits_dict in safe_load(lblp).items():
            limits_by_lang[lang] = ResourceLimits(**limits_dict)

    return limits_by_lang


if __name__ == "__main__":
    cfg = load_config(Path("execution_engine/config.yaml"))
    print(cfg.supported_languages.keys())
