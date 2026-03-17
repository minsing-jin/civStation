import logging
import sys

from computer_use_test.utils.run_log_cache import start_run_log_session


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
