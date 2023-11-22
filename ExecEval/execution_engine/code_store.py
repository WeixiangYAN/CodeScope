import os
import shutil
import uuid
from pathlib import Path

from config import CodeStoreConfig


class CodeStore:
    _source_dir: Path

    def __init__(self, cfg: CodeStoreConfig, run_ids: tuple[int, int]) -> None:
        self._source_dir = cfg.source_code_dir / uuid.uuid4().hex
        self.uid = run_ids[1]
        self.gid = run_ids[0]

    def create(self):
        os.makedirs(self._source_dir, exist_ok=True)
        os.chown(self._source_dir, self.uid, self.gid)
        os.chmod(self._source_dir, 0o775)

    def destroy(self) -> None:
        shutil.rmtree(self._source_dir, ignore_errors=True)

    def write_source_code(self, source_code: str, filename: Path) -> Path:
        filepath = self._source_dir / filename

        with filepath.open("w") as fp:
            fp.write(source_code)

        filepath = filepath.resolve()

        os.chown(filepath, self.uid, self.gid)
        os.chmod(filepath, 0o775)
        return filepath

    def read_source_code(self, filepath: Path) -> str:
        with filepath.open() as f:
            s = f.read()

        return s


if __name__ == "__main__":
    from config import load_config

    cfg = load_config(Path("execution_engine/config.yaml"))
    code_store = CodeStore(cfg.code_store)
    print(
        code_store.read_source_code(
            code_store.write_source_code("""print("Hello")""", Path("main.py"))
        )
    )
