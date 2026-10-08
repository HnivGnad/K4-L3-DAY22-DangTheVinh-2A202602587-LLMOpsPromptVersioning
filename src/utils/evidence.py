"""Lưu stdout UTF-8 từ lần chạy thật."""
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

class Tee:
    """Ghi đồng thời vào console và tệp."""
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()

@contextmanager
def evidence_log(filename):
    """Chỉ tạo log khi bước tương ứng thực sự chạy."""
    path = ROOT / "evidence" / filename
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        with redirect_stdout(Tee(sys.stdout, stream)):
            yield