#!/usr/bin/env python3
"""Launch and inspect runs using atomic, run-specific completion records."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def write_record(path, record):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
            temporary = output.name
            json.dump(record, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def read_record(path, run_id):
    with path.open() as source:
        record = json.load(source)
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("unsupported run status schema")
    if record.get("run_id") != run_id:
        raise ValueError("status belongs to a different run")
    return record


def run(args):
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    path = args.status.resolve()
    record = {
        "schema_version": 1,
        "run_id": args.run_id,
        "launcher_pid": os.getpid(),
        "child_pid": None,
        "status": "running",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "ended_at": None,
        "child_exit_code": None,
        "last_successful_epoch": None,
        "artifacts": [],
        "diagnostics": "not_run",
        "completion_reason": "completed",
    }
    with locked(path):
        if path.exists():
            raise ValueError("status path already exists; use a new path for each run")
        write_record(path, record)

    child = None
    received_signal = None

    def forward_signal(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    previous_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    environment = dict(os.environ, EXPHIL_RUN_ID=args.run_id, EXPHIL_RUN_STATUS=str(path))
    try:
        if received_signal is not None:
            exit_code = -received_signal
        else:
            child = subprocess.Popen(command, env=environment, start_new_session=True)
            with locked(path):
                record = read_record(path, args.run_id)
                record["child_pid"] = child.pid
                write_record(path, record)
            if received_signal is not None:
                forward_signal(received_signal, None)
            exit_code = child.wait()
    except OSError as error:
        exit_code = 127
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                child.wait()
        print(f"run-status: could not launch command: {error}", file=sys.stderr)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    with locked(path):
        record = read_record(path, args.run_id)
        if received_signal is not None or exit_code < 0:
            status = "interrupted"
        elif exit_code != 0:
            status = "failed"
        elif record["diagnostics"] == "failed":
            status = "diagnostics_failed"
        else:
            status = record["completion_reason"]
        record.update(
            status=status, child_exit_code=exit_code, ended_at=timestamp(), updated_at=timestamp()
        )
        write_record(path, record)
    if received_signal is not None:
        return 128 + received_signal
    if exit_code < 0:
        return 128 - exit_code
    return exit_code or (1 if status == "diagnostics_failed" else 0)


def update(args):
    with locked(args.status):
        record = read_record(args.status, args.run_id)
        if record.get("status") != "running":
            raise ValueError("cannot update a finished run")
        if args.epoch is not None:
            previous = record["last_successful_epoch"]
            if args.epoch < 0 or (previous is not None and args.epoch < previous):
                raise ValueError("last successful epoch must not decrease or be negative")
            record["last_successful_epoch"] = args.epoch
        if args.artifact:
            record["artifacts"] = sorted(set(record["artifacts"] + args.artifact))
        if args.diagnostics:
            if record["diagnostics"] == "failed" and args.diagnostics != "failed":
                raise ValueError("a diagnostic failure cannot be cleared within a run")
            record["diagnostics"] = args.diagnostics
        if args.early_stopped:
            record["completion_reason"] = "early_stopped"
        record["updated_at"] = timestamp()
        write_record(args.status, record)
    return 0


def check(args):
    record = read_record(args.status, args.run_id)
    if args.pid is not None and record.get("launcher_pid") != args.pid:
        raise ValueError("status belongs to a different launcher PID")
    status = record.get("status")
    exit_code = record.get("child_exit_code")
    terminal = {"completed", "early_stopped", "failed", "interrupted", "diagnostics_failed"}
    if not isinstance(status, str) or status not in terminal or type(exit_code) is not int:
        raise ValueError("no verified terminal status; completion is unknown")
    ended_at = record.get("ended_at")
    if not isinstance(ended_at, str) or datetime.fromisoformat(ended_at).tzinfo is None:
        raise ValueError("missing or invalid completion time")
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, str) or diagnostics not in {"not_run", "passed", "failed"}:
        raise ValueError("invalid diagnostic status")
    if status in {"completed", "early_stopped"}:
        if exit_code != 0 or record["diagnostics"] == "failed":
            raise ValueError("inconsistent successful status")
    print(f"{status}: run={args.run_id}, child_exit_code={exit_code}, diagnostics={diagnostics}")
    return 0 if status in {"completed", "early_stopped"} else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation, function in (("run", run), ("update", update), ("check", check)):
        subparser = commands.add_parser(operation)
        subparser.set_defaults(function=function)
        subparser.add_argument("--status", type=Path, default=os.environ.get("EXPHIL_RUN_STATUS"))
        subparser.add_argument("--run-id", default=os.environ.get("EXPHIL_RUN_ID"))
        if operation == "run":
            subparser.add_argument("command", nargs=argparse.REMAINDER)
        elif operation == "update":
            subparser.add_argument("--epoch", type=int)
            subparser.add_argument("--artifact", action="append", default=[])
            subparser.add_argument("--diagnostics", choices=["passed", "failed"])
            subparser.add_argument("--early-stopped", action="store_true")
        else:
            subparser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if not args.status or not args.run_id or not args.run_id.strip():
        parser.error("--status and a nonempty --run-id (or EXPHIL_RUN_STATUS/EXPHIL_RUN_ID) are required")
    try:
        return args.function(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"run-status: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
