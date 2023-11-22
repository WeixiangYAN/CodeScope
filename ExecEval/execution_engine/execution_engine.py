import os
import shlex
import signal
import subprocess
from pathlib import Path
from threading import Timer
from unittest import ExtendedUnittest

import gmpy2
from code_store import CodeStore
from config import Config
from exec_outcome import ExecOutcome
from helper import convert_crlf_to_lf
from job import JobData, LanguageError
from prlimit import get_prlimit_str
from resource_limit import ResourceLimits
from runtime import Runtime
from seccomp_filter import make_filter
from settings import JavaClassNotFoundError


class CompilationError(Exception):
    """Shows the compilation error message

    Args:
        Exception command list[str]: command to compile
        message str: compilation error message
    """

    def __init__(self, command, message: subprocess.CalledProcessError):
        self.command = command
        self.message = message
        super().__init__(f"command: {self.command} produced: {self.message.stderr}")


def init_validate_outputs():
    _token_set = {"yes", "no", "true", "false"}
    PRECISION = gmpy2.mpfr(1e-12, 129)

    def validate_outputs(output1: str, output2: str) -> bool:
        # for space sensitive problems stripped string should match
        def validate_lines(lines1, lines2):
            validate_line = lambda lines: lines[0].strip() == lines[1].strip()
            if len(lines1) != len(lines2):
                return False
            return all(map(validate_line, zip(lines1, lines2)))

        if validate_lines(output1.strip().split("\n"), output2.strip().split("\n")):
            return True

        # lines didn't work so token matching
        tokens1, tokens2 = output1.strip().split(), output2.strip().split()
        if len(tokens1) != len(tokens2):
            return False

        for tok1, tok2 in zip(tokens1, tokens2):
            try:
                num1, num2 = gmpy2.mpfr(tok1, 129), gmpy2.mpfr(tok2, 129)
                if abs(num1 - num2) > PRECISION:
                    return False
            except ValueError:
                if tok1.lower() in _token_set:
                    tok1 = tok1.lower()
                if tok2.lower() in _token_set:
                    tok2 = tok2.lower()
                if tok1 != tok2:
                    return False

        return True

    return validate_outputs


class ExecutionEngine:
    def __init__(
        self,
        cfg: Config,
        limits_by_lang: dict[str, ResourceLimits],
        run_ids: tuple[int, int],
        logger,
    ) -> None:
        self.code_store = CodeStore(cfg.code_store, run_ids)
        self.supported_languages: dict[str, Runtime] = dict()
        self.output_validator = init_validate_outputs()
        for lang, sup_cfg in cfg.supported_languages.items():
            self.supported_languages[lang] = Runtime(sup_cfg)

        self.run_uid = run_ids[1]
        self.run_gid = run_ids[0]
        self.socket_filter = make_filter(["socket"])
        self.logger = logger
        self.limits_by_lang = limits_by_lang

        self.exec_env = os.environ.copy()
        self.exec_env["GOCACHE"] = str(self.code_store._source_dir.resolve())

    def start(self):
        self.code_store.create()

    def stop(self):
        self.code_store.destroy()

    def _compile(self, command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            shlex.split(command),
            user=self.run_uid,
            group=self.run_gid,
            capture_output=True,
            cwd=self.code_store._source_dir,
            env=self.exec_env,
            timeout=20,
        )

    def _get_executable_after_compile(
        self,
        lang: str,
        source_file: Path,
        cmd: str | None = None,
        flags: str | None = None,
    ) -> tuple[str | Path, bool]:
        if not self.supported_languages[lang].is_compiled_language:
            return source_file, False

        compile_str, executable = self.supported_languages[lang].compile(
            source_file, cmd, flags
        )

        cp = self._compile(compile_str)
        if cp.returncode == 0:
            return executable, False

        return cp.stderr.decode(errors="ignore"), True

    def get_executor(self, job: JobData, limits: ResourceLimits) -> tuple[str | Path | LanguageError, int]:
        language = job.language
        if language is None:
            return LanguageError("Language must be selected to execute a code."), -1

        if language not in self.supported_languages:
            return LanguageError(f"Support for {language} is not implemented."), -1

        source_code = convert_crlf_to_lf(job.source_code)

        if self.supported_languages[language].has_sanitizer and job.use_sanitizer:
            source_code = self.supported_languages[language].sanitize(source_code)

        source_path = self.supported_languages[language].get_file_path(source_code)
        if isinstance(source_path, JavaClassNotFoundError):
            return source_path, -1
        source_path = self.code_store.write_source_code(source_code, source_path)

        executable, err = self._get_executable_after_compile(
            language, source_path, cmd=job.compile_cmd, flags=job.compile_flags
        )

        if err:
            return executable, -1

        execute_flags = job.execute_flags
        
        if self.supported_languages[language].extend_mem_for_vm:
            if limits._as != -1:
                if execute_flags is None:
                    execute_flags = f" -{self.supported_languages[language].extend_mem_flag_name}{limits._as} "
                else:
                    execute_flags += f" -{self.supported_languages[language].extend_mem_flag_name}{limits._as} "

        return (
            self.supported_languages[language].execute(
                executable, cmd=job.execute_cmd, flags=execute_flags
            ),
            self.supported_languages[language].timelimit_factor,
        )

    def check_output_match(self, job: JobData) -> list[ExtendedUnittest]:
        limits = job.limits
        if limits is None:
            limits = ResourceLimits()
            limits.update(self.limits_by_lang[job.language])
            
        executor, timelimit_factor = self.get_executor(job, limits)
        # raise CompilationError(e.args, e)
        if timelimit_factor == -1:
            result = executor
            if isinstance(executor, (LanguageError, JavaClassNotFoundError)):
                result = executor.msg
            elif not isinstance(result, str):
                result = "Some bug in ExecEval, please do report."
            return [
                ExtendedUnittest(
                    input="",
                    output=[],
                    result=result,
                    exec_outcome=ExecOutcome.COMPILATION_ERROR,
                )
            ]

        # if language uses vm then add extra 1gb smemory for the parent vm program to run
        if self.supported_languages[job.language].extend_mem_for_vm and limits._as != -1:
            limits._as += 2**30
        # executor = f"timeout -k {limits.cpu} -s 9 {limits.cpu * timelimit_factor + 0.5} {get_prlimit_str(limits)} {executor}"
        executor = f"{get_prlimit_str(limits)} {executor}"
        new_test_cases = job.unittests.copy()
        self.logger.debug(
            f"Execute with gid={self.run_gid}, uid={self.run_uid}: {executor}"
        )
        for key, tc in enumerate(job.unittests):
            result, exec_outcome = None, None
            outs, errs = None, None
            syscall_filter_loaded = False

            def preexec_fn():
                nonlocal syscall_filter_loaded
                if job.block_network:
                    self.socket_filter.load()
                    syscall_filter_loaded = True

            with subprocess.Popen(
                shlex.split(executor),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                user=self.run_uid,
                group=self.run_gid,
                preexec_fn=preexec_fn,
                cwd=self.code_store._source_dir.resolve(),
                env=self.exec_env,
                start_new_session=True,
            ) as child_process:

                def handler():
                    if child_process.poll() is None:
                        child_process.kill()

                timer = Timer(limits.cpu * timelimit_factor + 1, handler)
                timer.start()
                # self.logger.debug(f"PID: {child_process.pid}")
                try:
                    outs, errs = child_process.communicate(
                        tc.input.encode("ascii"), timeout=limits.cpu * timelimit_factor
                    )
                    timer.cancel()
                except subprocess.TimeoutExpired:
                    exec_outcome = ExecOutcome.TIME_LIMIT_EXCEEDED
                except subprocess.CalledProcessError:
                    exec_outcome = ExecOutcome.RUNTIME_ERROR
                    if errs is not None:
                        result = errs.decode(errors="ignore").strip()
                finally:
                    timer.cancel()
                    child_process.kill()
                    child_process.communicate()
                    child_process.wait()
                    if syscall_filter_loaded:
                        self.socket_filter.reset()
                if exec_outcome is None:
                    if child_process.returncode == 0 and outs is not None:
                        result = outs.decode(errors="ignore").strip()
                        exec_outcome = (
                            ExecOutcome.PASSED
                            if any(
                                self.output_validator(output, result)
                                for output in tc.output
                            )
                            else ExecOutcome.WRONG_ANSWER
                        )
                    elif errs is not None and len(errs) != 0:
                        exec_outcome = ExecOutcome.RUNTIME_ERROR
                        errs = errs.decode(errors="ignore")
                        if "out of memory" in errs.lower():
                            exec_outcome = ExecOutcome.MEMORY_LIMIT_EXCEEDED
                        if child_process.returncode > 0:
                            result = errs
                        else:
                            result = f"Process exited with code {-child_process.returncode}, {signal.strsignal(-child_process.returncode)} stderr: {errs}"
                    else:
                        exec_outcome = ExecOutcome.MEMORY_LIMIT_EXCEEDED
                        if outs is not None:
                            result = outs.decode(errors="ignore").strip()
                        elif errs is not None:
                            result = errs.decode(errors="ignore").strip()
                        else:
                            self.logger.debug("**************** MEMORY_LIMIT_EXCEEDED assigned but no stdout or stderr")
            new_test_cases[key].update_result(result)
            new_test_cases[key].update_exec_outcome(exec_outcome)
            if job.stop_on_first_fail and exec_outcome is not ExecOutcome.PASSED:
                break

        return new_test_cases


if __name__ == "__main__":

    class Test:
        file: str
        lang: str

        def __init__(self, file, lang):
            self.file = file
            self.lang = lang

    tests = [
        Test("execution_engine/test_codes/test.c", "GNU C"),
        Test("execution_engine/test_codes/test.cpp", "GNU C++17"),
        Test("execution_engine/test_codes/test.go", "Go"),
        Test("execution_engine/test_codes/test.js", "Node js"),
        Test("execution_engine/test_codes/test.php", "PHP"),
        Test("execution_engine/test_codes/test.py", "PyPy 3"),
        Test("execution_engine/test_codes/test.py", "Python 3"),
        Test("execution_engine/test_codes/test.rb", "Ruby"),
        Test("execution_engine/test_codes/test.rs", "Rust"),
        Test("execution_engine/test_codes/test.java", "Java 7"),
        Test("execution_engine/test_codes/test.kt", "Kotlin"),
    ]

    unittests = [
        ExtendedUnittest("1 1", ["2"]),
        ExtendedUnittest("1 3", ["4"]),
        ExtendedUnittest("-1 2", ["1"]),
        ExtendedUnittest("122 2", ["124"]),
    ]

    from config import load_config
    from job import JobData
    from resource_limit import ResourceLimits

    cfg = load_config(Path("execution_engine/config.yaml"))

    ce = ExecutionEngine(cfg)

    for t in tests:
        with open(t.file) as f:
            s = f.read()
        updated_unittests = ce.check_output_match(
            JobData(
                language=t.lang,
                source_code=s,
                unittests=unittests,
                limits=ResourceLimits(),
            )
        )

        print(f"{t.lang} got: \n", updaed_unittests)
