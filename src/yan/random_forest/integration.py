"""Adapter from the common scheduler to Yan's RF implementation.

Only this adapter publishes the three shared RF files. Native checkpoints and
diagnostics stay in a run-specific subdirectory. Existing shared CSVs are never
accepted as evidence that a Yan run is complete.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
import sys
import time
import uuid

import numpy as np
import pandas as pd

from .audit import (ROOT, MODEL, audit, data_and_splits, digest, environment, sha,
                    write_json, git)
from .train import (run_fold, validate_fold, load_checkpoint, checkpoint_path,
                    export_and_verify, utc)

IMPLEMENTATION = "yan.random_forest.train.run_fold"
MARKER = "random_forest_export.json"
DETAIL = Path("yan/random_forest")
POINTER = "active_run.json"
SHARED_FILES = ("predictions/Random_Forest.csv",
                "outer_fold_metrics_Random_Forest.csv",
                "inner_search_Random_Forest.csv")
SOURCES = ("src/final_analysis.py", "src/data_prep.py", "src/metrics.py",
           "src/yan/random_forest/train.py", "src/yan/random_forest/audit.py",
           "src/yan/random_forest/integration.py",
           "src/yan/random_forest/__init__.py", "requirements.txt",
           "src/yan/random_forest/requirements.txt")


def identity(data, repeats):
    """Fit identity excludes mutable README text, git commit labels and scores."""
    from final_analysis import MODEL_GRIDS
    config = dict(model=MODEL, repeats=repeats, outer_folds=5, inner_folds=5,
        n_estimators=500, grid=MODEL_GRIDS[MODEL], n_jobs=1, outer_task_workers=2,
        target_sensitivity=.95, outer_seed="707+r", inner_seed="1707+100*r+f",
        candidate_seed="inner_seed+100*c+k", final_seed="707+10000*r+100*f",
        tie_rule="largest mean inner AUROC; earliest grid candidate on exact tie")
    return dict(config=config, environment=environment(),
        source_sha256={p: sha(ROOT / p) for p in SOURCES},
        data_sha256={p: sha(ROOT / p) for p in
                     ("data/raw/wdbc.data", "data/processed/wdbc_clean.csv", "707-proposal.md")},
        ordered_ids=data.id.astype(int).tolist(),
        ordered_labels=data.target.astype(int).tolist())


def _native_path(output, relative):
    native = (output / relative).resolve()
    if not native.is_relative_to((output / DETAIL / "runs").resolve()):
        raise ValueError("RF manifest points outside its own run directory")
    return native


def validate_frames(predictions, folds, searches, data, outer, repeats):
    expected = outer.loc[outer["repeat"].le(repeats)].sort_values(["repeat", "id"]).reset_index(drop=True)
    p = predictions.sort_values(["repeat", "id"]).reset_index(drop=True)
    if len(p) != 569*repeats or p.duplicated(["model", "repeat", "id"]).any():
        raise ValueError("Shared RF predictions are incomplete or duplicated")
    if set(p.model) != {MODEL}:
        raise ValueError("Shared RF model name changed")
    pd.testing.assert_frame_equal(p[["id", "y_true", "repeat", "fold"]],
        expected[["id", "y_true", "repeat", "fold"]], check_dtype=False)
    if not np.isfinite(p[["y_prob", "threshold"]]).all().all():
        raise ValueError("Nonfinite shared RF probability/threshold")
    if not p.y_prob.between(0, 1).all() or not p.threshold.between(0, 1).all():
        raise ValueError("Shared RF probability/threshold outside [0,1]")
    if not p.groupby(["repeat", "fold"]).threshold.nunique().eq(1).all():
        raise ValueError("Shared RF thresholds vary within outer fold")
    if not np.array_equal(p.predicted_class, p.y_prob.ge(p.threshold).astype(int)):
        raise ValueError("Shared RF predicted_class disagrees with locked threshold")
    keys = {(r, f) for r in range(1, repeats+1) for f in range(5)}
    for frame in [folds, searches]:
        if set(frame.model) != {MODEL} or set(zip(frame["repeat"], frame.fold)) != keys:
            raise ValueError("Shared RF search/fold keys are incompatible")
    if len(folds) != 5*repeats or folds.duplicated(["repeat", "fold"]).any():
        raise ValueError("Shared RF outer records are incomplete/duplicated")
    if len(searches) != 80*repeats or searches.duplicated(["repeat", "fold", "candidate"]).any():
        raise ValueError("Shared RF candidate records are incomplete/duplicated")
    if not searches.groupby(["repeat", "fold"]).candidate.apply(
            lambda c: set(c) == set(range(16))).all():
        raise ValueError("Candidate indices must remain 0-15")
    for row in folds.to_dict("records"):
        r, f = row["repeat"], row["fold"]
        g = p.loc[p["repeat"].eq(r) & p.fold.eq(f)]
        selected = searches.loc[searches["repeat"].eq(r) & searches.fold.eq(f)]
        selected = selected.sort_values(["mean_inner_auroc", "candidate"], ascending=[False, True]).iloc[0]
        if row["selected_candidate"] != selected.candidate or row["params_json"] != selected.params_json:
            raise ValueError("Selected candidate/search records do not agree")
        if not np.isclose(row["threshold"], g.threshold.iloc[0], atol=1e-14, rtol=0):
            raise ValueError("Fold and prediction thresholds do not agree")
        if row["n_validation"] != len(g) or row["n_train"] != len(data)-len(g):
            raise ValueError("Shared RF cohort sizes do not agree")
    return digest(p[["id", "y_true", "repeat", "fold"]].to_dict("records"))


def validate_shared_export(output):
    """A commit marker and all three hashes are required for integrated runs.

    An untouched legacy archive without an adapter pointer/marker is still
    readable by summarize, which does not claim a new training run occurred.
    """
    output = Path(output).resolve()
    pointer_path = output / DETAIL / POINTER
    marker_path = output / MARKER
    if not pointer_path.exists() and not marker_path.exists():
        return None
    if not pointer_path.exists() or not marker_path.exists():
        raise ValueError("RF training/export is incomplete; refusing stale shared RF files")
    pointer = json.loads(pointer_path.read_text())
    marker = json.loads(marker_path.read_text())
    native = _native_path(output, pointer["run_directory"])
    manifest = json.loads((native / "run_manifest.json").read_text())
    if manifest["status"] != "complete_verified":
        raise ValueError("Active RF run is not fully exported; resume it before summarizing")
    if not (marker["run_id"] == manifest["run_id"] == pointer["run_id"] and
            marker["fingerprint"] == manifest["fingerprint"]):
        raise ValueError("Shared RF files belong to a different/incomplete run")
    if marker["implementation"] != IMPLEMENTATION or set(marker["file_sha256"]) != set(SHARED_FILES):
        raise ValueError("Wrong RF implementation or incomplete export bundle")
    for name, expected in marker["file_sha256"].items():
        if not (output / name).exists() or sha(output / name) != expected:
            raise ValueError(f"Mixed or modified shared RF export: {name}")
    data, outer, _ = data_and_splits()
    frames = [pd.read_csv(output / name) for name in SHARED_FILES]
    key_hash = validate_frames(*frames, data, outer, manifest["config"]["repeats"])
    if key_hash != marker["id_label_fold_sha256"]:
        raise ValueError("Shared RF ID/label/repeat/fold mapping changed")
    return marker


def publish(output, native, manifest, data, outer):
    """Stage/validate all files, back up previous slots, then commit a marker.

    If a process stops between the three replacements, the old/new hashes and
    active-run ID will not pass validate_shared_export. A resume repairs the
    entire set from validated native checkpoints; no old rows are appended.
    """
    stage = native / "shared_export"
    stage.mkdir(exist_ok=True)
    p = pd.read_csv(native / "oof_predictions.csv")
    folds = pd.read_csv(native / "fold_results.csv")
    searches = pd.read_csv(native / "inner_cv_results.csv")
    folds["seconds"] = folds.total_seconds
    for frame in [folds, searches]:
        frame["convergence_warnings"] = frame.warnings_json.map(
            lambda s: sum(str(v).startswith("ConvergenceWarning:") for v in json.loads(s)))
    frames = [p, folds, searches]
    for frame, name in zip(frames, SHARED_FILES):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
    reread = [pd.read_csv(stage / name) for name in SHARED_FILES]
    key_hash = validate_frames(*reread, data, outer, manifest["config"]["repeats"])
    hashes = {name: sha(stage / name) for name in SHARED_FILES}
    backup = native / "previous_shared_files"
    for name in [*SHARED_FILES, MARKER]:
        old = output / name
        if old.exists() and not (backup / name).exists():
            (backup / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, backup / name)
    # The native staged copies stay available as evidence. Replace complete
    # files; never update a subset of rows in an old RF CSV.
    for name in SHARED_FILES:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".csv.rf-new")
        shutil.copy2(stage / name, tmp)
        tmp.replace(target)
    marker = dict(run_id=manifest["run_id"], fingerprint=manifest["fingerprint"],
        implementation=IMPLEMENTATION, created_utc=utc(),
        detail_run=str(native.relative_to(output)), file_sha256=hashes,
        id_label_fold_sha256=key_hash, repeats=manifest["config"]["repeats"],
        outer_folds=5, inner_folds=5, n_estimators=500,
        prediction_rows=len(p), outer_records=len(folds), candidate_records=len(searches),
        formal_ten_repeat_analysis=manifest["config"]["repeats"] == 10,
        summary_status="pending")
    write_json(output / MARKER, marker)


def record_shared_summary(output):
    """Record which RF run the newly rebuilt common tables contain."""
    output = Path(output).resolve()
    marker = validate_shared_export(output)
    if marker is None:
        return
    combined = pd.read_csv(output / "outer_predictions_all_models.csv")
    repeat = pd.read_csv(output / "repeat_metrics.csv")
    summary = pd.read_csv(output / "model_summary.csv")
    if len(combined.loc[combined.model.eq(MODEL)]) != marker["prediction_rows"]:
        raise ValueError("Common summary includes missing/duplicate RF predictions")
    if repeat.model.eq(MODEL).sum() != marker["repeats"] or summary.model.eq(MODEL).sum() != 1:
        raise ValueError("Common summary must contain exactly one Random_Forest")
    names = ["outer_predictions_all_models.csv", "repeat_metrics.csv", "model_summary.csv"]
    marker["summary_sha256"] = {name: sha(output / name) for name in names}
    marker["summary_status"] = "complete"
    marker["summary_rf_run_id"] = marker["run_id"]
    write_json(output / MARKER, marker)


def run_shared_random_forest(data, output, repeats=10, outer_folds=5, inner_folds=5, resume=True):
    """Run/resume Yan RF once for the common scheduler and export shared files."""
    if outer_folds != 5 or inner_folds != 5 or not 1 <= repeats <= 10:
        raise ValueError("RF requires five outer/inner folds and 1-10 repeat prefixes; formal analysis is 10 repeats")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checked_data, outer, inner = data_and_splits()
    pd.testing.assert_frame_equal(data.reset_index(drop=True), checked_data, check_dtype=False)
    fit_identity = identity(data, repeats)
    fingerprint = digest(fit_identity)
    pointer_path = output / DETAIL / POINTER
    if resume and pointer_path.exists():
        pointer = json.loads(pointer_path.read_text())
        native = _native_path(output, pointer["run_directory"])
        manifest = json.loads((native / "run_manifest.json").read_text())
        if pointer["run_id"] != manifest["run_id"] or manifest["fingerprint"] != fingerprint:
            raise ValueError("RF checkpoint data/code/config/environment changed; use --no-resume with a new output directory")
    else:
        run_id = str(uuid.uuid4())
        native = output / DETAIL / "runs" / run_id
        native.mkdir(parents=True)
        info = audit(native)
        manifest = dict(run_id=run_id, fingerprint=fingerprint, fit_identity=fit_identity,
            config=fit_identity["config"], audit=info, status="initialized", started_utc=utc(),
            implementation=IMPLEMENTATION, base_commit=git("rev-parse", "HEAD"),
            sessions=[], commands=[], source_changes="Shared RF dispatch/export adapter; statistical RF loop unchanged",
            formal_analysis_complete=False, old_shared_csvs_used_as_checkpoints=False)
        write_json(native / "run_manifest.json", manifest)
        pointer = dict(run_id=run_id, run_directory=str(native.relative_to(output)))
        write_json(pointer_path, pointer)
    targets = [(r, f) for r in range(1, repeats+1) for f in range(5)]
    pending = []
    for r, f in targets:
        path = checkpoint_path(native, r, f)
        if path.exists():
            load_checkpoint(path, manifest, r, f, data, outer, inner)
        else:
            pending.append((r, f))
    reused = len(targets)-len(pending)
    print(f"Random_Forest -> {IMPLEMENTATION}; {len(pending)} outer folds require fitting; "
          f"resume={resume}; diagnostics={native}", flush=True)
    tick = time.perf_counter()
    fitted = 0
    manifest["status"] = "running"
    manifest["commands"].append(dict(utc=utc(), argv=sys.argv))
    write_json(native / "run_manifest.json", manifest)

    def save(payload, r, f):
        nonlocal fitted
        validate_fold(payload, r, f, data, outer, inner)
        write_json(checkpoint_path(native, r, f), dict(run_id=manifest["run_id"],
            fingerprint=fingerprint, payload_sha256=digest(payload), payload=payload))
        fitted += 1
        print(f"Yan RF checkpoint: repeat={r}, fold={f}, candidates=16, inner fits=80, final fits=1", flush=True)

    try:
        # First full outer fold checks the same code path; it is not run twice.
        if (1, 0) in pending:
            save(run_fold(1, 0), 1, 0)
            pending.remove((1, 0))
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(run_fold, r, f): (r, f) for r, f in pending}
            for future in as_completed(futures):
                r, f = futures[future]
                save(future.result(), r, f)
        if digest(identity(data, repeats)) != fingerprint:
            raise RuntimeError("RF source/config/data/environment changed during training")
        checks = export_and_verify(native, manifest, complete=repeats == 10)
        checks.update(planned_outer_folds=len(targets), planned_run_complete=True,
                      formal_ten_repeat_analysis=repeats == 10)
        write_json(native / "verification.json", checks)
        manifest["status"] = "exporting"
        write_json(native / "run_manifest.json", manifest)
        publish(output, native, manifest, data, outer)
        manifest["status"] = "complete_verified"
        manifest["formal_analysis_complete"] = repeats == 10
        manifest["verification"] = checks
    except BaseException as error:
        manifest["status"] = "interrupted_or_failed"
        manifest["last_error"] = repr(error)
        raise
    finally:
        manifest["sessions"].append(dict(stage="run", entrypoint="src/final_analysis.py nested",
            finished_utc=utc(), wall_seconds=time.perf_counter()-tick,
            newly_fitted_outer_folds=fitted, checkpoint_folds_reused=reused))
        write_json(native / "run_manifest.json", manifest)
    marker = validate_shared_export(output)
    print(f"RF shared export verified: {marker['prediction_rows']} predictions, "
          f"{marker['outer_records']} outer rows, {marker['candidate_records']} candidates", flush=True)
    return marker
