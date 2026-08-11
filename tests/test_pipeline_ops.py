"""Tests for tracking, provenance, search and the training supervisor.

These guard the operational guarantees, which fail silently by nature: a
tracker that quietly logs nothing, a token that ends up in ``.git/config``, a
supervisor that observes the test split or ignores its own budget.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amdt.experiments.supervisor import (METHOD_ALTERING_KEYS, PatchStub, Supervisor,
                                         TripConfig)
from amdt.utils.repo import (GITATTRIBUTES, GITIGNORE, GitRepo, load_pat,
                             provenance_message)
from amdt.utils.tracking import NullTracker, build_tracker, config_hash

HAS_GIT = shutil.which("git") is not None


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #
def test_null_tracker_always_leaves_a_record(tmp_path):
    t = NullTracker(tmp_path)
    t.log({"loss": 0.5}, step=1)
    t.log({"loss": 0.4}, step=2)
    t.summary({"best": 0.4})
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["_step"] == 2
    assert json.loads((tmp_path / "summary.json").read_text())["best"] == 0.4


def test_build_tracker_falls_back_when_disabled(tmp_path):
    class Cfg(dict):
        __getattr__ = dict.get
    cfg = Cfg(tracking=Cfg(enabled=False))
    t = build_tracker(cfg, tmp_path, seed=0)
    assert isinstance(t, NullTracker)
    t.log({"x": 1})           # must not raise
    t.finish()


def test_config_hash_is_stable_and_sensitive():
    a = {"lr": 0.001, "seed": 0}
    b = {"seed": 0, "lr": 0.001}          # key order must not matter
    c = {"lr": 0.002, "seed": 0}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert len(config_hash(a)) == 7


# --------------------------------------------------------------------------- #
# provenance / git
# --------------------------------------------------------------------------- #
def test_provenance_message_carries_everything_needed_to_trace_a_number():
    msg = provenance_message("best: val_f1 0.8412 (prev 0.8377) @ epoch 37",
                             seed=42, config_hash="a3f9c1e", run_id="3kx9m2p1",
                             trigger="new_best_val", agent_revision="2/3")
    for token in ("seed: 42", "config_hash: a3f9c1e", "wandb_run: 3kx9m2p1",
                  "trigger: new_best_val", "agent_revision: 2/3"):
        assert token in msg
    assert msg.splitlines()[1] == ""       # blank line after the subject


@pytest.mark.skipif(not HAS_GIT, reason="git not installed")
def test_repo_init_writes_policy_and_commits(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    r = GitRepo(tmp_path)
    r.init(lfs=False)
    assert (tmp_path / ".gitignore").read_text() == GITIGNORE
    sha = r.commit(["src"], provenance_message("init", 0, "abc1234"))
    assert sha and len(sha) == 40
    assert not r.is_dirty()


@pytest.mark.skipif(not HAS_GIT, reason="git not installed")
def test_repo_commit_is_a_noop_when_nothing_changed(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    r = GitRepo(tmp_path)
    r.init(lfs=False)
    r.commit(["src"], "first")
    assert r.commit(["src"], "second") is None       # no diff -> no commit


@pytest.mark.skipif(not HAS_GIT, reason="git not installed")
def test_token_in_git_config_is_detected(tmp_path):
    r = GitRepo(tmp_path)
    r.init(lfs=False)
    cfg = tmp_path / ".git" / "config"
    cfg.write_text(cfg.read_text() +
                   '\n[remote "leak"]\n\turl = https://ghp_' + "a" * 36 +
                   "@github.com/o/r.git\n")
    with pytest.raises(RuntimeError, match="token is embedded"):
        r.assert_no_token_in_config()


def test_gitignore_excludes_weights_but_keeps_best_model():
    assert "*.pt" in GITIGNORE and "!best_model.pt" in GITIGNORE
    assert "outputs/" in GITIGNORE and "wandb/" in GITIGNORE
    assert "best_model.pt filter=lfs" in GITATTRIBUTES


def test_load_pat_reads_environment_only(monkeypatch):
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    assert load_pat() is None
    monkeypatch.setenv("GITHUB_PAT", "ghp_" + "b" * 36)
    assert load_pat().startswith("ghp_")


@pytest.mark.skipif(not HAS_GIT, reason="git not installed")
def test_push_without_a_token_declines_rather_than_guessing(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    r = GitRepo(tmp_path, remote="https://github.com/o/r.git")
    r.init(lfs=False)
    assert r.push() is False


# --------------------------------------------------------------------------- #
# supervisor
# --------------------------------------------------------------------------- #
def _sup(tmp_path, **kw):
    return Supervisor(tmp_path, TripConfig(min_epochs_between=0, **kw))


def test_supervisor_refuses_to_observe_the_test_split(tmp_path):
    s = _sup(tmp_path)
    with pytest.raises(AssertionError, match="never observe the test split"):
        s.observe(1, val=0.3, test_pe=0.29)


def test_divergence_fires_immediately(tmp_path):
    s = _sup(tmp_path)
    for e in range(6):
        s.observe(e, val=0.4, train_loss=0.5)
    s.observe(6, val=0.4, train_loss=float("nan"))
    assert s.check()["trigger"] == "divergence"


def test_loss_spike_fires_divergence(tmp_path):
    s = _sup(tmp_path)
    for e in range(6):
        s.observe(e, val=0.4, train_loss=0.5)
    s.observe(6, val=0.4, train_loss=50.0)
    assert s.check()["trigger"] == "divergence"


def test_plateau_fires_and_a_healthy_run_does_not(tmp_path):
    s = _sup(tmp_path, patience=4, min_delta=0.01)
    for e in range(8):
        s.observe(e, val=0.40, train_loss=0.5)
    assert s.check()["trigger"] == "plateau"

    s2 = _sup(tmp_path / "b", patience=4, min_delta=0.01)
    for e in range(8):
        s2.observe(e, val=0.40 - 0.02 * e, train_loss=0.5)
    assert s2.check() is None


def test_underperformance_respects_warmup(tmp_path):
    s = _sup(tmp_path, baseline_metric=0.30, warmup_epochs=5, patience=100)
    s.observe(1, val=0.49, train_loss=0.5)
    assert s.check() is None                      # still warming up
    s.observe(9, val=0.49, train_loss=0.5)
    assert s.check()["trigger"] == "underperformance"


def test_budget_is_enforced(tmp_path):
    s = _sup(tmp_path, patience=2, min_delta=0.01, max_rounds=3)
    fired = 0
    for e in range(60):
        s.observe(e, val=0.40, train_loss=0.5)
        # vary the current lr so each proposal differs and thrash suppression
        # does not mask the budget being tested
        if s.maybe_intervene({"lr": 1e-3 * (0.9 ** e), "weight_decay": 5e-4 * (1.1 ** e)}):
            fired += 1
    assert fired == 3
    assert s.rounds_used == 3
    assert len(s.interventions()) == 3


def test_identical_patches_are_suppressed(tmp_path):
    s = _sup(tmp_path, patience=2, min_delta=0.01, max_rounds=10)
    for e in range(10):
        s.observe(e, val=0.40, train_loss=0.5)
    first = s.maybe_intervene({"lr": 1e-3, "weight_decay": 5e-4})
    assert first is not None
    for e in range(10, 20):
        s.observe(e, val=0.40, train_loss=0.5)
    # same inputs -> same proposal -> must be suppressed, not re-applied
    again = s.maybe_intervene({"lr": 1e-3, "weight_decay": 5e-4})
    assert again is None or again.patch != first.patch


def test_min_epochs_between_prevents_thrash(tmp_path):
    s = Supervisor(tmp_path, TripConfig(patience=2, min_delta=0.01,
                                        min_epochs_between=10, max_rounds=5))
    for e in range(6):
        s.observe(e, val=0.4, train_loss=0.5)
    assert s.maybe_intervene({"lr": 1e-3}) is not None
    s.observe(7, val=0.4, train_loss=0.5)
    assert s.maybe_intervene({"lr": 5e-4}) is None      # too soon


def test_method_altering_patches_are_tagged_and_surfaced(tmp_path):
    s = Supervisor(tmp_path)
    assert s.categorise({"lr": 1e-4}) == "mechanical"
    assert s.categorise({"architecture": "wider"}) == "method_altering"
    assert all(k in METHOD_ALTERING_KEYS for k in ("loss", "architecture", "model"))


def test_intervention_log_is_jsonl_and_complete(tmp_path):
    s = _sup(tmp_path, patience=2, min_delta=0.01)
    for e in range(6):
        s.observe(e, val=0.4, train_loss=0.5)
    iv = s.maybe_intervene({"lr": 1e-3}, seed=7, config_hash="deadbee")
    assert iv is not None
    rec = json.loads((tmp_path / "agent_interventions.jsonl").read_text().splitlines()[0])
    for k in ("ts", "round", "trigger", "rule", "category", "files",
              "diff_summary", "val_before"):
        assert k in rec


# --------------------------------------------------------------------------- #
# patch stub
# --------------------------------------------------------------------------- #
class _FakeOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": 1e-3, "weight_decay": 5e-4}]


def test_stub_applies_a_patch_once_and_records_it(tmp_path):
    (tmp_path / "agent_patch.json").write_text(json.dumps(
        {"round": 1, "epoch": 5, "patch": {"lr": 2e-4}, "category": "mechanical",
         "trigger": "plateau"}))
    stub = PatchStub(tmp_path, tmp_path, max_rounds=3)
    opt = _FakeOptimizer()

    entry = stub.poll_and_apply(epoch=5, val_pe=0.4, train_loss=0.5, optimizer=opt)
    assert entry["applied"]["lr"] == 2e-4
    assert opt.param_groups[0]["lr"] == 2e-4
    # same round again -> no-op
    assert stub.poll_and_apply(epoch=6, val_pe=0.4, train_loss=0.5, optimizer=opt) is None


def test_stub_defers_what_it_cannot_apply_in_place(tmp_path):
    (tmp_path / "agent_patch.json").write_text(json.dumps(
        {"round": 1, "patch": {"lr": 1e-4, "optimizer": "sgd"}}))
    stub = PatchStub(tmp_path, tmp_path)
    entry = stub.poll_and_apply(epoch=1, val_pe=0.4, train_loss=0.5,
                                optimizer=_FakeOptimizer())
    assert "lr" in entry["applied"] and "optimizer" in entry["deferred"]


def test_stub_honours_its_round_cap(tmp_path):
    stub = PatchStub(tmp_path, tmp_path, max_rounds=1)
    stub.rounds_used = 1
    (tmp_path / "agent_patch.json").write_text(json.dumps({"round": 2, "patch": {"lr": 1}}))
    assert stub.poll_and_apply(epoch=1, val_pe=0.4, train_loss=0.5,
                               optimizer=_FakeOptimizer()) is None


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def test_search_objectives_reject_a_test_split():
    from amdt.experiments.search import _assert_no_test
    _assert_no_test(val_idx=[1, 2])                     # fine
    with pytest.raises(AssertionError, match="never see the test split"):
        _assert_no_test(test_idx=[3])
    with pytest.raises(AssertionError):
        _assert_no_test(X_test=[3])


# --------------------------------------------------------------------------- #
# W&B preflight: verify auth without ever surfacing the key
# --------------------------------------------------------------------------- #
class _Cfg(dict):
    __getattr__ = dict.get


def _tracking_cfg(**kw):
    base = {"enabled": True, "project": "GA-Stegno", "entity": None}
    base.update(kw)
    return _Cfg(tracking=_Cfg(**base))


def test_preflight_reports_missing_wandb_without_raising(monkeypatch):
    import builtins
    from amdt.utils.tracking import preflight

    real_import = builtins.__import__

    def no_wandb(name, *a, **k):
        if name == "wandb":
            raise ImportError("no module named wandb")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_wandb)
    pf = preflight(_tracking_cfg())
    assert pf["wandb_installed"] is False
    assert pf["ok"] is False
    assert "wandb login" in pf["hint"]


def test_preflight_finds_an_environment_credential_but_never_returns_it(monkeypatch):
    pytest.importorskip("wandb")
    from amdt.utils.tracking import preflight

    secret = "abc123def456" * 4
    monkeypatch.setenv("WANDB_API_KEY", secret)
    pf = preflight(_tracking_cfg())
    assert pf["credential_found"] is True
    assert pf["credential_source"] == "environment"
    # The key must not leak into the record that gets written to disk.
    assert secret not in json.dumps(pf)


def test_preflight_carries_the_project_name_through():
    from amdt.utils.tracking import preflight
    pf = preflight(_tracking_cfg(project="GA-Stegno"))
    assert pf["project"] == "GA-Stegno"


def test_preflight_prefers_a_configured_entity_over_the_default():
    from amdt.utils.tracking import preflight
    pf = preflight(_tracking_cfg(entity="some-team"))
    assert pf["configured_entity"] == "some-team"


def test_preflight_is_safe_when_tracking_is_absent():
    from amdt.utils.tracking import preflight
    pf = preflight(_Cfg())
    assert pf["enabled"] is False and pf["ok"] is False


def test_selecting_the_wandb_config_group_actually_enables_it():
    """`tracking=wandb` must not resolve to a disabled tracker.

    Regression guard: the group was shipped with enabled=false, so choosing it
    passed preflight (credential found, entity resolved) and then logged to
    JSONL anyway. A tracking config that is selected but inert is worse than no
    tracking, because the preflight reports success.
    """
    import yaml
    cfg_dir = Path(__file__).resolve().parents[1] / "configs" / "tracking"
    on = yaml.safe_load((cfg_dir / "wandb.yaml").read_text())
    off = yaml.safe_load((cfg_dir / "none.yaml").read_text())
    assert on["enabled"] is True, "tracking=wandb must enable tracking"
    assert off["enabled"] is False, "tracking=none must disable it"
    assert on["project"] == "GA-Stegno"
    assert on["entity"] == "yazan-aljeroudi-rachis-systems"
    assert not on["entity"].endswith("-org"), (
        "that is the organization; W&B refuses runs addressed to an org")
