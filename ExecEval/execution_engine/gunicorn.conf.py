import logging
import os

gunicorn_logger = logging.getLogger("gunicorn.error")


def clear_pid_from_worker_cfg_db(worker):
    worker_cfg_db = os.environ["WORKER_CFG_DB"]

    cfg_db_lines = []
    with open(worker_cfg_db) as db_rp:
        assigned = False
        for line in db_rp:
            pid, idx, gid, uid = map(int, line.strip().split(","))
            if not assigned and pid == worker.pid:
                assigned = True
                cfg_db_lines.append(",".join(map(str, (-1, idx, gid, uid))))
                gunicorn_logger.info(f"Remove {gid=} {uid=} from {pid=}")
            else:
                cfg_db_lines.append(line.strip())

    with open(worker_cfg_db, "w") as db_wp:
        for line in cfg_db_lines:
            db_wp.write(line + "\n")


def worker_abort(worker):
    clear_pid_from_worker_cfg_db(worker)

    if not hasattr(worker, "wsgi"):
        worker.wsgi = worker.app.wsgi()
    if hasattr(worker.wsgi, "config"):
        config = worker.wsgi.config
        if "execution_engine" in config:
            worker.wsgi.logger.info("Stopping execution_engine")
            config["execution_engine"].stop()


def worker_exit(server, worker):
    worker_abort(worker)


def when_ready(server):
    pass


def on_starting(server):
    run_gid_start = int(os.environ["RUN_GID"])
    run_uid_start = int(os.environ["RUN_UID"])
    num_workers = int(os.environ["NUM_WORKERS"])
    worker_cfg_db = os.environ["WORKER_CFG_DB"]

    with open(worker_cfg_db, "w") as db_wp:
        for i in range(num_workers):
            db_wp.write(f"-1,{i},{run_gid_start + i},{run_uid_start + i}\n")

    gunicorn_logger.info("Init worker cfg db.")


def pre_fork(server, worker):
    pass


def post_fork(server, worker):
    pass


def post_worker_init(worker):
    pass
