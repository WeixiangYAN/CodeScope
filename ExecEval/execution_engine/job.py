from dataclasses import dataclass, field
from unittest import ExtendedUnittest

from resource_limit import ResourceLimits

def fix_uts(uts):
	uts_fx = []
	for ut in uts:
		uts_fx.append({
			"input": ut["input"],
			"output": ut["output"],
		})
	return uts_fx

@dataclass
class JobData:
    language: str
    source_code: str
    unittests: list[ExtendedUnittest]
    compile_cmd: str | None = None
    compile_flags: str | None = None
    execute_cmd: str | None = None
    execute_flags: str | None = None
    limits: ResourceLimits | None = None
    block_network: bool = True
    stop_on_first_fail: bool = True
    use_sanitizer: bool = False

    @classmethod
    def json_parser(cls, form):
        return cls(
            language=form.get("language"),
            source_code=form.get("source_code"),
            unittests=[ExtendedUnittest(**t) for t in fix_uts(form.get("unittests"))],
            compile_cmd=form.get("compile_cmd"),
            compile_flags=form.get("compile_flags"),
            execute_cmd=form.get("execute_cmd"),
            execute_flags=form.get("execute_flags"),
            limits=ResourceLimits(**form.get("limits")) if form.get("limits") is not None else None,
            block_network=form.get("block_network", True),
            stop_on_first_fail=form.get("stop_on_first_fail", True),
            use_sanitizer=form.get("use_sanitizer", False),
        )


@dataclass
class LanguageError:
    msg: str
