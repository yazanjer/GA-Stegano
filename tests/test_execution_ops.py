"""Tests for execution targets, hardware labelling, dual-target supervision
and Kaggle acquisition (§12 dual-target, §13, §14).

These guard the failures that produce a *plausible but wrong* number: a timing
measured on a different VM, an `n_jobs=-1` that ties results to a core count, a
budget that resets on migration, a dataset that quietly gained rows.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amdt.data.kaggle import (DatasetPin, load_credentials, sha256_file,
                              verify_pin, warn_leaderboard_split)
from amdt.evaluation.tables import runtime_table
from amdt.experiments.registry_runs import (MigrationPolicy, PressureMonitor,
                                            RunRegistry, migrate_checkpoint)
from amdt.experiments.supervisor import Supervisor, TripConfig
from amdt.utils.execution import (ExecutionTarget, MPSRefused, configure_execution,
                                  hardware_fingerprint, memory_pressure,
                                  resolve_n_jobs)
from amdt.utils.profiling import StageProfiler


# --------------------------------------------------------------------------- #
# execution target (§13)
# --------------------------------------------------------------------------- #
def test_local_target_pins_threads_and_records_them():
    t = configure_execution(name="local", device="cpu", reported=True, threads=1)
    assert t.thread_mode == "pinned"
    assert t.threads == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert t.timing_valid


def test_exploration_mode_leaves_threads_free_and_says_so():
    t = configure_execution(name="local", device="cpu", reported=False, n_jobs=4)
    assert t.thread_mode == "multithreaded"
    # metrics are fine; the run just must not contribute timings to a table
    assert t.n_jobs == 4


def test_hosted_cpu_timings_are_marked_invalid():
    t = configure_execution(name="colab_cpu", device="cpu", reported=False)
    assert t.timing_valid is False
    assert "hosted" in (t.timing_invalid_reason or "").lower()


def test_mps_is_refused_for_reported_runs():
    with pytest.raises(MPSRefused, match="refused for reported runs"):
        configure_execution(name="local", device="mps", reported=True)


def test_mps_allowed_only_when_labelled_and_then_timing_invalid():
    t = configure_execution(name="local", device="mps", reported=True, allow_mps=True)
    assert t.timing_valid is False
    assert "MPS" in (t.timing_invalid_reason or "")


def test_n_jobs_minus_one_is_rejected():
    t = ExecutionTarget(reported=False, thread_mode="multithreaded", n_jobs=3,
                        hardware={"memory_total_gb": 64})
    assert resolve_n_jobs(-1, t) == 3          # falls back to the explicit value
    assert resolve_n_jobs(4, t) == 4


def test_pinned_reported_run_forces_single_worker():
    t = ExecutionTarget(reported=True, thread_mode="pinned", n_jobs=1,
                        hardware={"memory_total_gb": 64})
    assert resolve_n_jobs(8, t) == 1           # BLAS reduction order must be fixed


def test_low_ram_host_caps_workers():
    t = ExecutionTarget(reported=False, thread_mode="multithreaded", n_jobs=8,
                        hardware={"memory_total_gb": 16})
    assert resolve_n_jobs(8, t) == 2           # joblib copies the data per worker


def test_hardware_fingerprint_has_what_a_table_needs():
    hw = hardware_fingerprint()
    for k in ("platform", "machine", "cpu_count_logical", "python", "is_colab"):
        assert k in hw


def test_memory_pressure_returns_numbers_not_opinions():
    r = memory_pressure()
    assert all(isinstance(v, (int, float)) for v in r.values())


# --------------------------------------------------------------------------- #
# hardware labelling in results (rows 21, 22, 27)
# --------------------------------------------------------------------------- #
def test_profiler_stamps_each_stage_with_its_hardware():
    t = ExecutionTarget(name="local", device="cpu", hardware={"cpu_model": "TestCPU"})
    p = StageProfiler(target=t)
    with p.stage("embed"):
        pass
    rows = p.summary()
    assert rows[0]["hardware"] == t.label
    assert rows[0]["timing_valid"] is True
    assert p.spans_devices is False


def test_same_stage_on_two_devices_is_flagged_mixed():
    a = ExecutionTarget(name="local", device="cpu", hardware={"cpu_model": "A"})
    b = ExecutionTarget(name="colab_gpu", device="cuda", hardware={"gpu_name": "T4"})
    p = StageProfiler(target=a)
    with p.stage("train"):
        pass
    p.target = b
    with p.stage("train"):
        pass
    row = p.summary()[0]
    assert row["hardware"] == "MIXED"
    assert row["timing_valid"] is False
    assert p.spans_devices is True


def test_runtime_table_adds_a_hardware_column_only_when_needed():
    same = [{"stage": "a", "n": 1, "wall_mean_s": 1.0, "wall_std_s": 0.0,
             "wall_total_s": 1.0, "hardware": "CPU: X", "timing_valid": True},
            {"stage": "b", "n": 1, "wall_mean_s": 2.0, "wall_std_s": 0.0,
             "wall_total_s": 2.0, "hardware": "CPU: X", "timing_valid": True}]
    tex = runtime_table(same)
    assert "Hardware" not in tex
    assert "All timings measured on" in tex

    mixed = [dict(same[0]), dict(same[1], hardware="GPU: T4", timing_valid=False,
                                 timing_invalid_reason="ran on a hosted VM")]
    tex2 = runtime_table(mixed)
    assert "Hardware" in tex2
    assert r"$^{\dagger}$" in tex2
    assert "hosted VM" in tex2


# --------------------------------------------------------------------------- #
# dual-target supervision and migration (§12)
# --------------------------------------------------------------------------- #
def test_registry_persists_the_budget_across_processes(tmp_path):
    path = tmp_path / "runs.json"
    r1 = RunRegistry(path)
    r1.register("exp-1", target="local")
    assert r1.consume_round("exp-1", 3) is True
    assert r1.consume_round("exp-1", 3) is True

    r2 = RunRegistry(path)                    # supervisor restarted
    assert r2.get("exp-1").rounds_used == 2
    assert r2.consume_round("exp-1", 3) is True
    assert r2.consume_round("exp-1", 3) is False       # budget spent
    assert r2.get("exp-1").status == "halted"


def test_budget_is_shared_across_targets_not_per_target(tmp_path):
    reg = RunRegistry(tmp_path / "runs.json")
    reg.register("exp-2", target="local")
    reg.consume_round("exp-2", 3)
    reg.consume_round("exp-2", 3)
    reg.mark_migrated("exp-2", "hosted", "load sustained")
    # migration must not hand the run a fresh budget
    assert reg.get("exp-2").rounds_used == 2
    assert reg.consume_round("exp-2", 3) is True
    assert reg.consume_round("exp-2", 3) is False


def test_migration_marks_timings_invalid_but_not_metrics(tmp_path):
    reg = RunRegistry(tmp_path / "runs.json")
    reg.register("exp-3", target="local")
    rec = reg.mark_migrated("exp-3", "hosted", "memory > 90% for 5 samples")
    assert rec.timing_valid is False
    assert "wall-clock" in rec.timing_invalid_reason
    assert rec.target == "hosted"
    assert rec.history[0]["from"] == "local"


def test_supervisor_budget_comes_from_the_registry(tmp_path):
    reg = RunRegistry(tmp_path / "runs.json")
    sup = Supervisor(tmp_path, TripConfig(patience=2, min_delta=0.01,
                                          min_epochs_between=0, max_rounds=2),
                     experiment_id="exp-4", registry=reg, target="local")
    fired = 0
    for e in range(40):
        sup.observe(e, val=0.4, train_loss=0.5)
        if sup.maybe_intervene({"lr": 1e-3 * 0.9 ** e, "weight_decay": 5e-4 * 1.1 ** e}):
            fired += 1
    assert fired == 2
    assert reg.get("exp-4").rounds_used == 2


def test_local_target_uses_the_relaunch_channel_not_a_branch(tmp_path):
    called = {}

    def relaunch(patch):
        called.update(patch)

    sup = Supervisor(tmp_path, TripConfig(patience=2, min_delta=0.01,
                                          min_epochs_between=0),
                     target="local", relaunch_fn=relaunch)
    for e in range(6):
        sup.observe(e, val=0.4, train_loss=0.5)
    iv = sup.maybe_intervene({"lr": 1e-3})
    assert iv is not None
    assert called                              # patched and relaunched in place
    rec = json.loads((tmp_path / "agent_patch.json").read_text())
    assert rec["target"] == "local"


def test_pressure_monitor_needs_a_sustained_breach(tmp_path):
    pol = MigrationPolicy(load_per_core=1.0, memory_used_fraction=0.8,
                          consecutive_samples=3, enabled=True)
    m = PressureMonitor(pol)
    assert m.sample({"load_per_core": 2.0}) is None      # 1/3
    assert m.sample({"load_per_core": 2.0}) is None      # 2/3
    reason = m.sample({"load_per_core": 2.0})            # 3/3
    assert reason and "load/core" in reason


def test_transient_spike_does_not_migrate(tmp_path):
    pol = MigrationPolicy(load_per_core=1.0, consecutive_samples=3, enabled=True)
    m = PressureMonitor(pol)
    m.sample({"load_per_core": 5.0})
    m.sample({"load_per_core": 0.1})            # recovered -> streak resets
    assert m.streak == 0
    assert m.sample({"load_per_core": 5.0}) is None


def test_migration_policy_disabled_never_fires():
    m = PressureMonitor(MigrationPolicy(enabled=False, consecutive_samples=1))
    assert m.sample({"load_per_core": 99.0, "memory_used_fraction": 1.0}) is None


def test_migrate_checkpoint_requires_rng_state(tmp_path):
    torch = pytest.importorskip("torch")
    src = tmp_path / "ck.pt"
    torch.save({"model_state_dict": {}, "optimizer_state_dict": {}, "epoch": 3},
               src)
    with pytest.raises(ValueError, match="rng_state"):
        migrate_checkpoint(src, tmp_path / "out" / "ck.pt")


# --------------------------------------------------------------------------- #
# Kaggle (§14)
# --------------------------------------------------------------------------- #
def test_credentials_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    c = load_credentials()
    assert c.source == "environment" and c.username == "someone"
    assert "secret" not in json.dumps(c.describe())     # key never surfaces


def test_loose_permissions_on_kaggle_json_are_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    cfgdir = tmp_path / ".kaggle"
    cfgdir.mkdir()
    kj = cfgdir / "kaggle.json"
    kj.write_text(json.dumps({"username": "u", "key": "k"}))
    kj.chmod(0o644)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(cfgdir))
    with pytest.raises(PermissionError, match="600"):
        load_credentials()


def test_missing_credentials_give_actionable_guidance(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError) as exc:
        load_credentials()
    msg = str(exc.value)
    assert "Secrets" in msg and "chmod 600" in msg
    assert "chat" in msg          # says where NOT to put the key


def test_version_drift_warns_and_does_not_upgrade():
    pinned = DatasetPin(slug="a/b", version="1", sha256="a" * 64)
    live = DatasetPin(slug="a/b", version="2", sha256="a" * 64)
    out = verify_pin(live, pinned)
    assert out.drift is True
    assert "NOT auto-upgraded" in out.drift_note
    assert out.version == "2"        # reported, not silently replaced


def test_hash_mismatch_is_drift_even_at_the_same_version():
    pinned = DatasetPin(slug="a/b", version="1", sha256="a" * 64)
    live = DatasetPin(slug="a/b", version="1", sha256="b" * 64)
    assert verify_pin(live, pinned).drift is True


def test_matching_pin_is_not_drift():
    pinned = DatasetPin(slug="a/b", version="1", sha256="a" * 64)
    assert verify_pin(DatasetPin(slug="a/b", version="1", sha256="a" * 64),
                      pinned).drift is False


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world" * 1000)
    assert sha256_file(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_pin_roundtrips_through_json(tmp_path):
    pin = DatasetPin(slug="a/b", version="3", sha256="c" * 64, license="CC-BY")
    pin.save(tmp_path / "pin.json")
    assert DatasetPin.load(tmp_path / "pin.json").as_dict() == pin.as_dict()


def test_leaderboard_warning_states_the_trap():
    msg = warn_leaderboard_split()
    assert "not a held-out test set" in msg.lower() or "NOT a held-out" in msg
    assert "carve your own test split" in msg.lower()


def test_gitignore_covers_credentials():
    from amdt.utils.repo import GITIGNORE
    for pattern in ("kaggle.json", ".env", "*.key"):
        assert pattern in GITIGNORE
