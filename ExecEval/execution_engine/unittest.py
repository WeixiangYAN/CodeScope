from dataclasses import dataclass, field

from exec_outcome import ExecOutcome
from helper import convert_crlf_to_lf


@dataclass
class Unittest:
    input: str
    output: str
    result: str | None = None
    exec_outcome: ExecOutcome | None = None

    def __post_init__(self):
        self.input = convert_crlf_to_lf(self.input)
        self.output = convert_crlf_to_lf(self.output)

    def update_result(self, result):
        self.result = result

    def update_exec_outcome(self, exec_outcome):
        self.exec_outcome = exec_outcome

    def match_output(self):
        return self.result == self.output


@dataclass
class ExtendedUnittest:
    input: str
    output: list[str] = field(default_factory=list)
    result: str | None = None
    exec_outcome: ExecOutcome | None = None

    def __post_init__(self):
        self.input = convert_crlf_to_lf(self.input)
        self.output = [convert_crlf_to_lf(o) for o in self.output.copy()]

    def update_result(self, result):
        self.result = result

    def update_exec_outcome(self, exec_outcome):
        self.exec_outcome = exec_outcome

    def match_output(self, result=None):
        if result is None:
            result = self.result
        return result in self.output

    def json(self):
        _json = self.__dict__.copy()
        if self.exec_outcome is not None:
            _json["exec_outcome"] = self.exec_outcome.value

        return _json
