"""Tests for automatic run-folder sync to Drive (§15, audit rows 28-30).

The failure mode this guards against is a sync that *looks* configured and
quietly does nothing, or one that uploads half-written files. Both produce a
Drive folder you cannot trust, which is worse than no Drive folder.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amdt.utils.drive_sync import (DEFAULT_EXCLUDES, DriveNotFound, DriveSync,
                                   SyncConfig, resolve_drive_root,
                                   runs_referenced_by_commits)


def _cfg(root: Path, **kw) -> SyncConfig:
    return SyncConfig(enabled=True, root=str(root), interval_s=1, **kw)


def _run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run" / "2026-01-01_00-00-00"
    (d / "tables").mkdir(parents=True)
    (d / "figures").mkdir()
    (d / "results").mkdir()
    (d / "tables" / "tab_quality.tex").write_text(r"\toprule")
    (d / "figures" / "fig02.pdf").write_bytes(b"%PDF-1.4 fake")
    (d / "results" / "quality.csv").write_text("a,b\n1,2\n")
    (d / "provenance.json").write_text("{}")
    (d / "run.log").write_text("started\n")
    return d


# --------------------------------------------------------------------------- #
# row 28: automatic, no prompt, does not wait for the run to finish
# --------------------------------------------------------------------------- #
def test_sync_starts_immediately_without_waiting_for_the_run(tmp_path):
    run = _run_dir(tmp_path)
    s = DriveSync(run, _cfg(tmp_path / "drive")).start()
    try:
        # start() performs the first pass synchronously -- a crash one second in
        # must not lose the logs.
        assert (s.dest / "tables" / "tab_quality.tex").exists()
        assert (s.dest / "run.log").read_text() == "started\n"
    finally:
        s.close(prune=False)


def test_incremental_pass_picks_up_later_writes(tmp_path):
    run = _run_dir(tmp_path)
    s = DriveSync(run, _cfg(tmp_path / "drive"))
    s.sync_once()
    (run / "results" / "significance.csv").write_text("p\n0.01\n")
    time.sleep(0.01)
    n = s.sync_once()
    assert n == 1
    assert (s.dest / "results" / "significance.csv").exists()


def test_unchanged_files_are_not_recopied(tmp_path):
    run = _run_dir(tmp_path)
    s = DriveSync(run, _cfg(tmp_path / "drive"))
    first = s.sync_once()
    assert first > 0
    assert s.sync_once() == 0


def test_disabled_sync_is_a_clean_noop(tmp_path):
    run = _run_dir(tmp_path)
    s = DriveSync(run, SyncConfig(enabled=False))
    assert s.sync_once() == 0
    assert s.dest is None
    assert s.describe()["enabled"] is False


def test_enabled_sync_with_no_drive_fails_loudly(monkeypatch, tmp_path):
    # No explicit root, no Colab mount, no Drive for Desktop folder.
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setitem(sys.modules, "google.colab", None)
    sys.modules.pop("google.colab", None)
    monkeypatch.setattr("amdt.utils.drive_sync.Path.exists",
                        lambda self: False if "drive" in str(self).lower() else
                        os.path.exists(str(self)))
    with pytest.raises(DriveNotFound):
        resolve_drive_root(None)


def test_explicit_root_with_missing_parent_is_rejected(tmp_path):
    with pytest.raises(DriveNotFound, match="does not exist"):
        resolve_drive_root(str(tmp_path / "no" / "such" / "parent" / "x"))


# --------------------------------------------------------------------------- #
# row 29: staged, then atomically moved
# --------------------------------------------------------------------------- #
def test_no_partial_files_are_left_behind(tmp_path):
    run = _run_dir(tmp_path)
    s = DriveSync(run, _cfg(tmp_path / "drive"))
    s.sync_once()
    assert list(s.dest.rglob("*.amdt-part")) == []


def test_staging_temp_lives_in_the_destination_filesystem(tmp_path, monkeypatch):
    """The temp must sit beside the target, or os.replace raises EXDEV."""
    run = _run_dir(tmp_path)
    s = DriveSync(run, _cfg(tmp_path / "drive"))
    seen = {}
    real_replace = os.replace

    def spy(src, dst):
        seen["src_parent"] = Path(src).parent
        seen["dst_parent"] = Path(dst).parent
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    s.sync_once()
    assert seen["src_parent"] == seen["dst_parent"]


def test_a_copy_failure_does_not_kill_the_run(tmp_path, monkeypatch):
    run = _run_dir(tmp_path)
    s = DriveSync(run, _cfg(tmp_path / "drive"))
    monkeypatch.setattr("amdt.utils.drive_sync.shutil.copy2",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    s.sync_once()                      # must not raise
    assert s.errors and "disk full" in s.errors[0]


# --------------------------------------------------------------------------- #
# row 30: exclusions
# --------------------------------------------------------------------------- #
def test_excluded_paths_never_cross(tmp_path):
    run = _run_dir(tmp_path)
    (run / ".git").mkdir()
    (run / ".git" / "config").write_text("[core]")
    (run / "wandb").mkdir()
    (run / "wandb" / "debug.log").write_text("x")
    (run / "stego").mkdir()
    (run / "stego" / "AMDT__0.1__0__0.png").write_bytes(b"png")
    (run / "big.pgm").write_bytes(b"P5")
    (run / "archive.zip").write_bytes(b"PK")
    (run / "partial.part").write_text("half")

    s = DriveSync(run, _cfg(tmp_path / "drive"))
    s.sync_once()

    for gone in (".git/config", "wandb/debug.log", "stego/AMDT__0.1__0__0.png",
                 "big.pgm", "archive.zip", "partial.part"):
        assert not (s.dest / gone).exists(), gone
    # and the things that matter did cross
    assert (s.dest / "tables" / "tab_quality.tex").exists()
    assert (s.dest / "provenance.json").exists()


def test_default_excludes_cover_the_named_categories():
    joined = " ".join(DEFAULT_EXCLUDES)
    for pat in (".git", "wandb", "*.pgm", "*.zip", ".venv/*"):
        assert pat in joined


def test_checkpoints_can_be_excluded(tmp_path):
    run = _run_dir(tmp_path)
    (run / "best_model.pt").write_bytes(b"weights")
    s = DriveSync(run, _cfg(tmp_path / "drive", include_checkpoints=False))
    s.sync_once()
    assert not (s.dest / "best_model.pt").exists()

    s2 = DriveSync(run, _cfg(tmp_path / "drive2", include_checkpoints=True))
    s2.sync_once()
    assert (s2.dest / "best_model.pt").exists()


def test_oversize_files_are_skipped(tmp_path):
    run = _run_dir(tmp_path)
    (run / "huge.bin").write_bytes(b"0" * 4096)
    s = DriveSync(run, _cfg(tmp_path / "drive", max_file_mb=0.001))
    s.sync_once()
    assert not (s.dest / "huge.bin").exists()


# --------------------------------------------------------------------------- #
# retention
# --------------------------------------------------------------------------- #
def test_prune_keeps_last_n_and_never_touches_evidence(tmp_path):
    drive = tmp_path / "drive"
    root = drive / "amdt-runs"
    root.mkdir(parents=True)
    for i in range(6):
        d = root / f"run-{i}"
        d.mkdir()
        (d / "x").write_text("x")
        os.utime(d, (1000 + i, 1000 + i))       # run-5 newest

    run = _run_dir(tmp_path)
    s = DriveSync(run, SyncConfig(enabled=True, root=str(root), keep_last_n=2))
    removed = s.prune(protected=["run-0"])

    assert (root / "run-0").exists()            # protected: named by a commit
    assert (root / "run-5").exists()            # within keep_last_n
    assert s.run_id not in removed              # never prunes the current run
    assert "run-1" in removed


def test_runs_referenced_by_commits_reads_the_provenance_lines(tmp_path):
    pytest.importorskip("subprocess")
    import shutil as _sh
    if not _sh.which("git"):
        pytest.skip("git not installed")
    from amdt.utils.repo import GitRepo, provenance_message

    r = GitRepo(tmp_path)
    r.init(lfs=False)
    (tmp_path / "f.txt").write_text("x")
    r.commit(["f.txt"], provenance_message(
        "results: 2026-01-01_00-00-00", seed=0, config_hash="abc",
        extra={"run_dir": "/tmp/outputs/amdt_rebuttal/2026-01-01_00-00-00"}))
    ids = runs_referenced_by_commits(r)
    assert "2026-01-01_00-00-00" in ids


def test_describe_carries_the_target_for_the_provenance_record(tmp_path):
    run = _run_dir(tmp_path)
    s = DriveSync(run, _cfg(tmp_path / "drive"))
    s.sync_once()
    d = s.describe()
    assert d["enabled"] and d["sync_target"] and d["run_id"] == run.name
    assert d["files_synced"] > 0
    assert "excludes" in d
