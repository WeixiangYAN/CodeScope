import logging
import os
import sys
import time
import traceback
from pathlib import Path

from config import load_config, load_limits_by_lang
from exec_outcome import ExecOutcome
from flask import Flask, request
from flask_cors import CORS
from job import JobData

sys.path.extend([str(Path(__file__).parent)])

from execution_engine import ExecutionEngine

app = Flask(__name__)
CORS(app)
config_path = Path("config.yaml")
cfg = load_config(config_path)
limits_by_lang_path = Path("limits_by_lang.yaml")
limits_by_lang = load_limits_by_lang(limits_by_lang_path)

gunicorn_logger = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

worker_cfg_db = os.environ["WORKER_CFG_DB"]

cfg_db_lines = []
run_ids = None
with open(worker_cfg_db) as db_rp:
    assigned = False
    for line in db_rp:
        pid, idx, gid, uid = map(int, line.strip().split(","))
        if not assigned and pid == -1:
            pid = os.getpid()
            assigned = True
            cfg_db_lines.append(",".join(map(str, (pid, idx, gid, uid))))
            run_ids = (gid, uid)
            app.logger.info(f"Assigned {gid=}, {uid=} to {pid=}")
        else:
            cfg_db_lines.append(line.strip())

with open(worker_cfg_db, "w") as db_wp:
    for line in cfg_db_lines:
        db_wp.write(line + "\n")

execution_engine = ExecutionEngine(cfg, limits_by_lang, run_ids, app.logger)
app.config["execution_engine"] = execution_engine
execution_engine.start()


@app.route("/api/execute_code", methods=["POST"])
def run_job():
    log, ret, st = "", None, time.perf_counter_ns()
    try:
        job = JobData.json_parser(request.json)
        log = f"api/execute_code: Executing for {job.language}"
        result = execution_engine.check_output_match(job)
        ret = {"data": [r.json() for r in result]}
        exec_outcomes = [
            r.exec_outcome
            for r in result
            if not (r.exec_outcome is None or r.exec_outcome is ExecOutcome.PASSED)
        ] + [ExecOutcome.PASSED]
        log = f"{log} time: {(time.perf_counter_ns()-st)/(1000_000_000)}s, |uts|={len(job.unittests)}, exec_outcome={exec_outcomes[0].value}"

    except Exception as e:
        ret = {"error": str(e) + f"\n{traceback.print_exc()}"}, 400
        log = f"{log} time: {(time.perf_counter_ns()-st)/(1000_000_000)}s, {ret}"
    app.logger.info(log)
    return ret


@app.route("/api/all_runtimes", methods=["GET"])
def all_runtimes():
    log, st = "", time.perf_counter_ns()
    runtimes = []
    for runtime in execution_engine.supported_languages.values():
        runtimes.append(runtime.get_info())
    ret = runtimes, 200
    log = f"api/all_runtimes: {log} time: {(time.perf_counter_ns()-st)/(1000_000_000)}s"

    app.logger.info(log)
    return ret


if __name__ == "__main__":
    app.run(host="0.0.0.0")
