"""Immutable physical-copy and code-archive helpers for the ISMRM analysis root."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from shlex import join as shlex_join
from typing import Iterable


R2_CASES = (
    "SAX100_2CH0_4CH0",
    "SAX100_2CH0_4CH100",
    "SAX100_2CH100_4CH0",
    "S2CH50_4CH100_fold0",
    "S2CH50_4CH100_fold1",
    "S2CH100_4CH50_fold0",
    "S2CH100_4CH50_fold1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash relative paths and bytes so a physical copy is auditable."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _write_json_new(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copytree_new(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"copy source is not a directory: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def copyfile_new(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"copy source is not a file: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def import_r2_subject(r2_root: Path, analysis_root: Path, subject: str) -> dict[str, object]:
    """Physically import only the r2 artifacts required by later analysis."""
    r2_root, analysis_root = r2_root.resolve(), analysis_root.resolve()
    copied: list[dict[str, str]] = []
    mapping = (
        (Path("sparse_inputs") / subject, Path("sparse_inputs") / subject, Path()),
        (Path("proposed") / subject, Path("reconstruction") / "proposed" / subject, Path()),
        (Path("fit_first") / subject, Path("reconstruction") / "fit_first" / subject, Path()),
        (Path("formal_endpoints/Q002S640") / subject, Path("endpoints") / "central" / "proposed" / subject, Path("central")),
        (Path("formal_endpoints/Fit-first") / subject, Path("endpoints") / "central" / "fit_first" / subject, Path("central")),
    )
    for case in R2_CASES:
        case_root = r2_root / subject / case
        if not case_root.is_dir():
            raise FileNotFoundError(f"missing r2 case root: {case_root}")
        for source_prefix, destination_prefix, suffix in mapping:
            source = case_root / source_prefix / case / suffix
            destination = analysis_root / destination_prefix / case
            copytree_new(source, destination)
            copied.append({
                "source": str(source),
                "destination": str(destination),
                "source_tree_sha256": tree_sha256(source),
                "destination_tree_sha256": tree_sha256(destination),
            })
        for source_name, destination in (
            ("reproduction_plan.json", analysis_root / "jobs" / "r2" / subject / f"{case}.reproduction_plan.json"),
            ("commands.log", analysis_root / "logs" / "r2" / f"{subject}.{case}.commands.log"),
        ):
            source = case_root / source_name
            copyfile_new(source, destination)
            copied.append({
                "source": str(source),
                "destination": str(destination),
                "source_sha256": _sha256(source),
                "destination_sha256": _sha256(destination),
            })
    record = {
        "schema": "ismrm_five_subject_r2_import/v1",
        "subject": subject,
        "source_r2_root": str(r2_root),
        "cases": list(R2_CASES),
        "copied": copied,
        "metrics": "intentionally_not_imported",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_new(analysis_root / "migrations" / "r2_import" / f"{subject}.json", record)
    return record


def archive_git_commit(repository: Path, commit: str, output: Path) -> dict[str, object]:
    """Create an immutable source tarball plus admission record from one commit."""
    repository, output = repository.resolve(), output.resolve()
    actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", commit], text=True).strip()
    if output.exists():
        record_path, archive = output / "COMMIT.json", output / "source.tar"
        if not record_path.is_file() or not archive.is_file():
            raise FileExistsError(f"refusing incomplete existing archive: {output}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("repository") != str(repository) or record.get("commit") != actual or record.get("archive_sha256") != _sha256(archive):
            raise FileExistsError(f"refusing mismatched existing archive: {output}")
        return record
    subject = subprocess.check_output(["git", "-C", str(repository), "show", "-s", "--format=%s", actual], text=True).strip()
    output.mkdir(parents=True)
    archive = output / "source.tar"
    subprocess.run(["git", "-C", str(repository), "archive", "--format=tar", "--output", str(archive), actual], check=True)
    record = {
        "schema": "ismrm_five_subject_git_archive/v1",
        "repository": str(repository),
        "commit": actual,
        "subject": subject,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_new(output / "COMMIT.json", record)
    return record


def execute_r3_plan(plan_path: Path, commands_log: Path) -> dict[str, object]:
    """Materialize a validated r3 plan; callers must use a preceding dry-run."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "cyj11_r3_reproduction_plan/v1":
        raise ValueError(f"unsupported r3 plan schema: {plan.get('schema')}")
    if commands_log.exists():
        raise FileExistsError(f"refusing overwrite: {commands_log}")
    created: list[dict[str, str]] = []
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    with commands_log.open("x", encoding="utf-8") as log:
        for step in plan["steps"]:
            kind = step.get("kind")
            if kind == "reuse_prepared":
                destination = Path(step["destination"])
                if not destination.is_dir():
                    raise FileNotFoundError(f"planned reusable prepared root is missing: {destination}")
                for stack, record in plan["source"]["observations"].items():
                    observation = destination / stack / "observations.npz"
                    if not observation.is_file() or _sha256(observation) != record["sha256"]:
                        raise ValueError(f"{stack}: prepared reuse SHA256 changed after r3 dry-run: {observation}")
                log.write(f"# {step['id']}\nverified prepared reuse {destination}\n")
                created.append({"id": step["id"], "destination": str(destination), "mode": "reused_after_sha256_verification"})
                continue
            if kind in {"copy_prepared", "copy_native_cache"}:
                source, destination = Path(step["source"]), Path(step["destination"])
                log.write(f"# {step['id']}\ncopytree {source} {destination}\n")
                copytree_new(source, destination)
                created.append({"id": step["id"], "source": str(source), "destination": str(destination), "tree_sha256": tree_sha256(destination)})
                continue
            argv = step.get("argv")
            if argv is None:
                raise ValueError(f"unsupported r3 plan step: {step}")
            command = [str(value) for value in argv]
            log.write(f"# {step['id']}\n{shlex_join(command)}\n")
            log.flush()
            subprocess.run(command, cwd=step["cwd"], check=True)
            created.append({"id": step["id"], "cwd": str(step["cwd"])})
    provenance_output = next(step["provenance_output"] for step in plan["steps"] if step.get("kind") == "copy_prepared")
    record = {
        "schema": "ismrm_five_subject_r3_generation/v1",
        "identity": plan["identity"],
        "historical_commit": plan["historical_commit"],
        "source": plan["source"],
        "provenance": plan["provenance"],
        "plan": str(plan_path.resolve()),
        "commands_log": str(commands_log.resolve()),
        "created": created,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_new(Path(provenance_output), record)
    return record


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    r2 = actions.add_parser("import-r2")
    r2.add_argument("--r2-root", required=True); r2.add_argument("--analysis-root", required=True); r2.add_argument("--subject", required=True)
    archive = actions.add_parser("archive-git")
    archive.add_argument("--repository", required=True); archive.add_argument("--commit", required=True); archive.add_argument("--output", required=True)
    execute = actions.add_parser("execute-r3-plan")
    execute.add_argument("--plan", required=True); execute.add_argument("--commands-log", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.action == "import-r2":
        payload = import_r2_subject(Path(args.r2_root), Path(args.analysis_root), args.subject)
    elif args.action == "archive-git":
        payload = archive_git_commit(Path(args.repository), args.commit, Path(args.output))
    else:
        payload = execute_r3_plan(Path(args.plan), Path(args.commands_log))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
