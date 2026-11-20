# Common-entry integration update

The preferred training entry is now src/final_analysis.py nested --models Random_Forest.
It dispatches to integration.py and this module, publishes all three shared RF files,
and runs common summarization once. See the root README and docs/RANDOM_FOREST_INTEGRATION.md.
The standalone instructions below remain available for personal audits.

# Random Forest personal review implementation

This is an RF-only fresh-run adapter to the unchanged shared analysis core.
The model is still named Random_Forest. The results/yan/random_forest directory
contains review and provenance artifacts awaiting team integration, not a second
formal model entry.

From the repository root, using Python 3.12:

    python -m pip install -r requirements.txt
    python -m pip install -r src/yan/random_forest/requirements.txt
    python scripts/yan/run.py audit
    python scripts/yan/run.py pilot --output results/yan/random_forest/reproduction
    python scripts/yan/run.py run --output results/yan/random_forest/reproduction
    python scripts/yan/run.py verify --output results/yan/random_forest/reproduction
    python scripts/yan/run.py report --output results/yan/random_forest/reproduction
    python src/yan/random_forest/verify.py --output results/yan/random_forest/reproduction

The delivered default output already contains the original cloud run.
Use a new subdirectory for a fresh reproduction on another machine. Identical
run fingerprints are required to reuse any checkpoint. The fingerprint covers
training code, shared code, raw/clean data, protocol, exact environment and
reference inputs. Cross-platform replays must use a new output directory.

The initial pilot uses 16 candidates x 5 inner folds x 500 trees plus a final
500-tree refit. It is reused as outer repeat 1, fold 0 in the full run after
validation. run completes the remaining 49 outer folds. Two outer tasks run
concurrently; every forest and native thread pool uses one thread.

Shared functions reused:

- final_analysis.load_data and _outer_split: ordering and outer CV;
- final_analysis.MODEL_GRIDS and make_model: fixed 16-setting grid and 500-tree RF;
- metrics.select_threshold_target_sensitivity: selected-inner-OOF threshold;
- final_analysis.summarize_prediction_rows: row thresholds and repeat pooling;
- metrics.compute_calibration_curve: per-repeat quantile-bin calibration.

The loop around these functions records additional diagnostics missing from
the shared runner. It preserves its inner splits, candidate/final seeds and
exact-tie selection rule. The only estimator execution setting changed is
n_jobs from -1 to 1, with outer-task parallelism instead.
It never calls the all-model driver, SVM importer or shared report builder.
It never reads archived RF predictions/metrics/models.

Additional checks, with the default delivered artifacts and without training:

    python src/yan/random_forest/verify.py
    python src/yan/random_forest/test_checks.py

The focused tests exercise ties, class-column lookup, mismatched labels/folds,
duplicate IDs, inner/outer isolation and checkpoints from a different run.
The full verifier also checks the 250 selected inner-fold AUROCs against the
exported selected OOF predictions and compares all checkpoint payloads with
their exported CSV records.

Integration: after review, replace the previous Random_Forest entry once in
the shared prediction collection. Do not append this as a second RF model or
mix old/new folds. No integration or publication is performed by this adapter.

Source data: Wolberg et al. (1993), Breast Cancer Wisconsin (Diagnostic), UCI,
https://doi.org/10.24432/C5DW2B , CC BY 4.0.
