import io
import logging
import sys
from pathlib import Path

from rich.console import Console

from computer_use_test.utils.run_log_cache import get_run_log_cache_path, start_run_log_session


def test_get_run_log_cache_path_defaults_to_project_tmp_root():
    expected = Path(__file__).resolve().parents[2] / ".tmp" / "computer_use_test" / "turn_runner_latest.log"
    assert get_run_log_cache_path() == expected


def test_run_log_session_writes_root_logger_records(tmp_path):
    session = start_run_log_session(base_dir=tmp_path)

    try:
        logger = logging.getLogger("tests.run_log_cache")
        logger.warning("raw log line")
    finally:
        session.close()

    contents = session.path.read_text(encoding="utf-8")
    assert "raw log line" in contents


def test_run_log_session_overwrites_previous_run_contents(tmp_path):
    first = start_run_log_session(base_dir=tmp_path)
    try:
        logging.getLogger("tests.run_log_cache").warning("first run only")
    finally:
        first.close()

    second = start_run_log_session(base_dir=tmp_path)
    try:
        logging.getLogger("tests.run_log_cache").warning("second run only")
    finally:
        second.close()

    assert first.path == second.path
    contents = second.path.read_text(encoding="utf-8")
    assert "second run only" in contents
    assert "first run only" not in contents


def test_run_log_session_writes_uncaught_traceback_and_restores_hook(tmp_path, monkeypatch):
    delegated = []

    def fake_previous_hook(exc_type, exc_value, exc_tb):
        delegated.append((exc_type, exc_value, exc_tb))

    monkeypatch.setattr(sys, "excepthook", fake_previous_hook)
    session = start_run_log_session(base_dir=tmp_path)

    try:
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)
    finally:
        session.close()

    contents = session.path.read_text(encoding="utf-8")
    assert "RuntimeError: boom" in contents
    assert delegated and delegated[0][0] is RuntimeError
    assert sys.excepthook is fake_previous_hook


def test_run_log_session_captures_stdout_stderr_and_restores_streams(tmp_path):
    previous_stdout = sys.stdout
    previous_stderr = sys.stderr
    session = start_run_log_session(base_dir=tmp_path)

    try:
        assert sys.stdout is not previous_stdout
        assert sys.stderr is not previous_stderr
        print("stdout line", flush=True)
        Console().print("rich line")
        sys.stderr.write("stderr line\n")
        sys.stderr.flush()
    finally:
        session.close()

    contents = session.path.read_text(encoding="utf-8")
    assert "stdout line" in contents
    assert "rich line" in contents
    assert "stderr line" in contents
    assert sys.stdout is previous_stdout
    assert sys.stderr is previous_stderr


def test_run_log_session_tolerates_closed_previous_stderr(tmp_path, monkeypatch):
    closed_stderr = io.StringIO()
    closed_stderr.close()
    monkeypatch.setattr(sys, "stderr", closed_stderr)

    session = start_run_log_session(base_dir=tmp_path)

    try:
        sys.stderr.write("stderr after close\n")
        sys.stderr.flush()
    finally:
        session.close()

    contents = session.path.read_text(encoding="utf-8")
    assert "stderr after close" in contents


def test_run_log_session_excepthook_still_logs_when_previous_stderr_is_closed(tmp_path, monkeypatch):
    closed_stderr = io.StringIO()
    closed_stderr.close()
    monkeypatch.setattr(sys, "stderr", closed_stderr)

    delegated = []

    def fake_previous_hook(exc_type, exc_value, exc_tb):
        delegated.append(exc_type)
        sys.stderr.write("delegated hook output\n")
        sys.stderr.flush()

    monkeypatch.setattr(sys, "excepthook", fake_previous_hook)
    session = start_run_log_session(base_dir=tmp_path)

    try:
        try:
            raise RuntimeError("boom with closed stderr")
        except RuntimeError as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)
    finally:
        session.close()

    contents = session.path.read_text(encoding="utf-8")
    assert "RuntimeError: boom with closed stderr" in contents
    assert "delegated hook output" in contents
    assert delegated == [RuntimeError]


def test_run_log_session_survives_uvicorn_logging_reconfiguration(tmp_path):
    from fastapi import FastAPI
    from uvicorn.config import Config

    session = start_run_log_session(base_dir=tmp_path)

    try:
        Config(FastAPI(), host="127.0.0.1", port=9999, log_level="warning")
        logging.getLogger("tests.run_log_cache").warning("logger after uvicorn config")
        print("stdout after uvicorn config", flush=True)
        sys.stderr.write("stderr after uvicorn config\n")
        sys.stderr.flush()
    finally:
        session.close()

    contents = session.path.read_text(encoding="utf-8")
    assert "logger after uvicorn config" in contents
    assert "stdout after uvicorn config" in contents
    assert "stderr after uvicorn config" in contents
