# QatarBird-AFM 2026: full Kaggle -> Qatar experiment package

This is the updated local-PC pipeline requested for the Qatar bird PhD work.

The critical change from the earlier starter is that the **complete Kaggle dataset is now the public training stage**. The 20-species subset is no longer used for the scientific pipeline.

## Research flow

```text
FULL KAGGLE DATASET
Sound Of 114 Species Of Birds Till 2022
D:\other\Bird\Dataset\CC0
        |
        |  full audit, metadata, duration, corruption, class balance
        v
recording-disjoint train / val / test
        |
        v
BirdMAE Base
DBD-research-group/Bird-MAE-Base
        |
        v
114-class supervised public bird training
        |
        +-- best BirdMAE encoder checkpoint
        +-- 114-class head
        +-- class prototypes
        +-- confusion matrix / per-class metrics
        |
        v
KAGGLE-TRAINED BIRD ENCODER
        |
        | transfer encoder parameters
        | exact scientific-name head transfer when possible
        v
QATAR RAW FIELD AUDIO
D:\other\Bird\Dataset\Qatar
        |
        +-- 20251014_STUDY ...
        +-- 20251016_STUDY ...
        +-- 20251019_STUDY ...
        +-- 20251022_STUDY ...
        +-- 20251024_STUDY ...
        |
        v
SDP-report BirdNET pseudo-labelling protocol
40-dB silence trim -> 5 s -> 3 s views at 0/1/2 s -> 2-of-3 confirmation
        |
        +-- clean one-target Qatar clips -> AP1/AP3
        +-- multi-label Qatar clips -> AP2
        |
        v
AP1 ACP-SAR -> shared QatarBird-AFM encoder
        |
        +-- AP2 UAPS polyphony decoder
        +-- AP3 DCGS confusion-graph specialists
```

## Public datasets and backbone

Kaggle dataset:

`https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022`

BirdMAE Base:

`https://huggingface.co/DBD-research-group/Bird-MAE-Base`

The public BirdMAE HF repository contains its custom model code and feature extractor. The code uses `trust_remote_code=True`. Review/trust that repository before running it.

## 1. Copy this project to your existing code directory

Recommended:

```text
D:\other\Bird\Code
```

Your dataset paths already match the default configuration:

```text
D:\other\Bird\Dataset\CC0
D:\other\Bird\Dataset\Qatar
```

If needed, edit only these fields in `configs\default.yaml`.

## 2. Fix the current PyTorch environment first

Your existing environment failed on `torch\lib\fbgemm.dll` with WinError 182. Do not use it for training.

```bat
conda deactivate
conda env remove -n qatarbird-afm -y
cd /d D:\other\Bird\Code
conda env create -f environment_gpu_windows.yml
conda activate qatarbird-afm-2026
python scripts\00_check_env.py
```

See `docs\WINDOWS_TORCH_FIX.md` if the DLL error continues.

CPU fallback:

```bat
conda env create -f environment_cpu.yml
conda activate qatarbird-afm-2026-cpu
python scripts\00_check_env.py
```

## 3. Probe BirdMAE before a long run

```bat
python scripts\00_probe_birdmae.py --config configs\default.yaml
```

The expected embedding dimension is 768.

## 4. FULL Kaggle dataset audit

This is mandatory before training.

```bat
python scripts\01_audit_kaggle.py --config configs\default.yaml
```

For SHA-256 duplicate analysis too:

```bat
python scripts\01_audit_kaggle.py --config configs\default.yaml --hash
```

Outputs:

```text
results\kaggle_analysis\dataset_summary.json
results\kaggle_analysis\recording_audit.csv
results\kaggle_analysis\species_statistics.csv
results\kaggle_analysis\class_distribution.png
results\kaggle_analysis\duration_distribution.png
```

Inspect `species_statistics.csv` before training. The script warns if the discovered folder count is not 114.

## 5. Build the full 114-class manifest

```bat
python scripts\02_build_kaggle_manifest.py --config configs\default.yaml
```

This creates:

```text
data\kaggle_114_recordings.csv
data\kaggle_114_clips.csv
data\kaggle_114_labels.csv
```

The split is made at original-recording level **before** 5-second clip expansion. Segments from one source recording therefore cannot appear in both train and test.

By default `max_clips_per_recording: 0`, so the scientific Kaggle stage uses **every complete 5-second clip from every readable recording**. Set a positive cap only for a deliberate development/smoke run, and do not report that capped run as the full-dataset result.

The script also tries to find the Kaggle metadata CSV and attach scientific names. If the Kaggle download on your PC does not contain that CSV, common-name folder labels still work, but classifier-head transfer to Qatar will be limited because exact scientific-name matching cannot be verified.

## 6. Train BirdMAE on the complete Kaggle dataset

```bat
python scripts\03_train_kaggle114.py --config configs\default.yaml
```

Default report-aligned settings:

```text
32,000 Hz
5-second clips
30 epochs max
batch size 16
learning rate 1e-4
one warm-up epoch
weight decay 0.01
gradient clipping 1.0
AMP
patience 5
```

The audio path can also apply the report's signal-processing chain:

```text
800 Hz - 15 kHz Butterworth band-pass
spectral gating using first 0.5 s noise estimate
RMS normalization to -20 dBFS
```

Saved public model:

```text
models\kaggle114\best.pt
models\kaggle114\classifier_head.pt
models\kaggle114\prototypes.pt
```

Evaluation output includes accuracy, balanced accuracy, macro-F1, weighted F1, macro precision/recall, Cohen kappa, MCC, log loss, AUROC where valid, average precision, top-3 accuracy, per-class metrics, confusion matrix, probabilities and embeddings.

You can rerun the test evaluation with:

```bat
python scripts\04_evaluate_kaggle114.py --config configs\default.yaml
```

## 7. Inspect Kaggle -> Qatar exact species transfer

After the Kaggle manifest exists:

```bat
python scripts\05_transfer_report.py --config configs\default.yaml
```

Output:

```text
results_summary\kaggle_to_qatar_species_mapping.csv
```

Only exact normalized scientific-name matches are used to copy classifier-head weights. The BirdMAE encoder itself is transferred regardless of direct species overlap.

## 8. Audit raw Qatar field recordings

```bat
python scripts\10_audit_qatar.py --config configs\default.yaml
```

Output:

```text
data\qatar_raw_audit.csv
```

This records study session, source filename, duration, sample rate and channels. Raw Qatar audio is never moved or overwritten.

## 9. Recreate the SDP report labelling preparation

The report states that BirdNET labelling used 40-dB silence removal, then 5-second segments. Run:

```bat
python scripts\11_prepare_qatar_label_clips.py --config configs\default.yaml
```

Derived clips go to:

```text
data\qatar_label_clips\
```

The code writes FLAC rather than uncompressed WAV to reduce disk use. Provenance remains in:

```text
data\qatar_label_clip_manifest.csv
```

See `docs\REPORT_LABELING_PROTOCOL.md` for the exact report-derived rules and the one threshold caveat.

## 10. Create the separate BirdNET environment

BirdNET uses a different runtime from the PyTorch research environment. Keep it isolated.

```bat
conda env create -f environment_birdnet_windows.yml
conda activate qatarbird-birdnet
```

Then run:

```bat
python scripts\12_birdnet_label_qatar.py --config configs\default.yaml --batch-clips 50
```

The script uses BirdNET 2.4 and reproduces the report's confirmation logic:

```text
5-second Qatar clip
   |-- 0-3 s
   |-- 1-4 s
   `-- 2-5 s

confirm target only if >=2 of 3 views meet threshold
```

Out-of-target BirdNET detections are ignored. Any BirdNET species in genus `Anthus` can map to the report's dataset-level `Pipit Spp.` target.

Default confidence is 0.60 because the report's overall validation Test 4 used 0.6. The report does not explicitly state one final production threshold for every local export, so this is intentionally configurable:

```bat
python scripts\12_birdnet_label_qatar.py --config configs\default.yaml --threshold 0.65
```

Do not change it without recording the change in the thesis.

Outputs:

```text
data\qatar_birdnet_windows.csv
data\qatar_confirmed_labels.csv
```

## 11. Build Qatar training manifests

Return to the PyTorch environment:

```bat
conda activate qatarbird-afm-2026
python scripts\13_build_qatar_manifests.py --config configs\default.yaml
```

Outputs:

```text
data\qatar_labels.csv
data\qatar_singlelabel_manifest.csv
data\qatar_multilabel_manifest.csv
data\qatar_dominant_manifest.csv
data\qatar_session_splits.csv
```

The default clean single-label AP1/AP3 training set uses only 5-second clips with exactly one confirmed target. It does not force an overlapping recording into one species by selecting the highest-confidence BirdNET label.

`qatar_dominant_manifest.csv` is retained only to reproduce/ablate the earlier dominant-voice method.

### Session split

With the five study folders visible in the Qatar dataset, the default is:

```text
first 3 sessions -> train
4th session      -> validation
5th session      -> test
```

The code also writes a leave-one-session-out fold table. This is materially stronger than random segment splitting.

## 12. 26-class report replication vs 27-class PhD experiment

The SDP report excluded Variable Wheatear because only about 5 minutes of public data were available.

Strict report mode is default:

```yaml
qatar_labeling:
  use_26_report_species: true
```

For the new 27-class PhD stress test:

```yaml
qatar_labeling:
  use_26_report_species: false
```

The full label/family mapping is in `configs\qatar_species_27.csv`. It contains 18 families.

## 13. AP1: Kaggle-trained BirdMAE -> Qatar ACP-SAR

Run the proposed full AP1:

```bat
python scripts\20_train_ap1.py --config configs\default.yaml --variant A5_full_acpsar
```

The code first loads:

```text
models\kaggle114\best.pt
```

into BirdMAE, then builds the Qatar adapter architecture.

If a Qatar scientific name exactly matches a Kaggle scientific name, the corresponding public 114-class classifier weight is also copied to the Qatar head. The transfer mapping is saved with the run.

AP1 variants:

```text
A0_frozen_linear
A1_single_adapter
A2_sparse_router
A3_router_condition
A4_router_prototype
A5_full_acpsar
```

Full AP1 combines environment-conditioned sparse adapters, condition-invariance loss, prototype preservation and router-balance regularization.

Stable checkpoint:

```text
models\qatar\ap1_best.pt
```

## 14. AP2: UAPS overlap/polyphony experiment

```bat
python scripts\21_train_ap2.py --config configs\default.yaml --variant B5_full_uaps
```

AP2 uses:

```text
5-s shared QatarBird-AFM representation
10-s context represented by two shared 5-s encoder passes
adaptive context gate
query set decoder
class predictions + clip-relative intervals
uncertainty-guided hard-pair curriculum
```

Training combines BirdNET pseudo-labelled Qatar segments with synthetic 2/3/4-species mixtures generated from clean Qatar clips. Synthetic mixtures carry exact interval labels. If a Qatar validation/test split is too small, the script explicitly falls back to a fixed synthetic evaluation and records this in `run_info.json` rather than hiding it.

AP2 automatically tunes one threshold per class on validation data, freezes those thresholds, then evaluates the test split.

Saved metrics include micro/macro/weighted F1, precision/recall, exact match, Hamming loss, Jaccard, average precision and cardinality error.

## 15. AP3: DCGS confusion-graph specialists

```bat
python scripts\22_train_ap3.py --config configs\default.yaml --variant C5_full_dcgs
```

The dynamic graph combines:

```text
empirical validation confusion
+ class prototype cosine similarity
+ uncertainty-weighted competitor scores
```

The model periodically rebuilds that graph and uses it to gate specialist corrections. The final variant adds graph-weighted contrastive separation and uncertainty-scaled correction.

Outputs include the learned graph, validation/test metrics, per-class results and all checkpoints.

## 16. Run all ablations

AP1 only:

```bat
python scripts\31_run_ablation_matrix.py --config configs\default.yaml --stage ap1
```

AP2 only:

```bat
python scripts\31_run_ablation_matrix.py --config configs\default.yaml --stage ap2
```

AP3 only:

```bat
python scripts\31_run_ablation_matrix.py --config configs\default.yaml --stage ap3
```

Everything:

```bat
python scripts\31_run_ablation_matrix.py --config configs\default.yaml --stage all
```

Use `--epochs 2` first if you only want to verify the execution path.

## 17. Multi-seed final runs

After architecture selection:

```bat
python scripts\32_run_multiseed.py --config configs\default.yaml --seeds 42 123 2026
```

For final thesis tables, compute and report mean +/- standard deviation across the selected seeds. Do not select a model on the test split.

## 18. Aggregate result tables

```bat
python scripts\30_aggregate_results.py --config configs\default.yaml
```

Outputs:

```text
results_summary\all_test_metrics.csv
results_summary\headline_metrics.csv
```

## 19. Recommended actual execution order on your PC

Do not start the complete ablation matrix immediately.

```text
1. recreate/fix conda environment
2. 00_check_env.py
3. 00_probe_birdmae.py
4. 01_audit_kaggle.py
5. inspect Kaggle audit CSVs/plots
6. 02_build_kaggle_manifest.py
7. 03_train_kaggle114.py
8. 10_audit_qatar.py
9. inspect Qatar raw audit
10. 11_prepare_qatar_label_clips.py
11. BirdNET environment + 12_birdnet_label_qatar.py
12. inspect pseudo-label counts manually
13. 13_build_qatar_manifests.py
14. 20_train_ap1.py A5
15. 21_train_ap2.py B5
16. 22_train_ap3.py C5
17. only then run ablations
18. final 3-seed selected-model experiment
```

## 20. Scientific caution about Qatar labels

The student SDP report used BirdNET as an automated labelling tool. These are therefore **model-assisted pseudo-labels**, not equivalent to an expert-annotated ground-truth field dataset.

For a PhD thesis, keep three concepts separate:

1. **Public Kaggle supervised labels**: folder/metadata labels used to train the public 114-class stage.
2. **Qatar BirdNET pseudo-labels**: automatically generated local labels following the SDP protocol.
3. **Expert-verified Qatar test subset**, if you can create one: this should be the preferred final field-performance benchmark.

If you obtain even a few hundred expert-reviewed Qatar clips, do not mix them into training immediately. Reserve an untouched expert test subset first.

## Qatar pseudo-label dataset analysis

After `13_build_qatar_manifests.py`, run:

```bat
python scripts\14_analyze_qatar_labels.py --config configs\default.yaml
```

It writes `results\qatar_analysis\` with per-species pseudo-labelled minutes, confidence statistics, session-by-species counts, label co-occurrence, label-cardinality distribution, and a direct comparison against the local-duration values reported in SDP Table 4-2. This analysis also records the report's stated total multi-label duration of `52:34:50`.

One report ambiguity is preserved rather than hidden: SDP Table 4-2 combines `Anthus` as “Anthus (Including Pipits)” and does not give a separate local duration for Long-billed Pipit. The PhD code keeps exact `Anthus similis` detections separate when BirdNET provides that scientific name; this choice is recorded in `configs\default.yaml` and should be reported in the methodology.

For a stricter replication of the SDP local `Anthus (Including Pipits)` handling, set:

```yaml
qatar_labeling:
  pipit_policy: combine_anthus_report
```

For the new experiment, leave `separate_if_exact` so an exact BirdNET `Anthus similis` result can remain Long-billed Pipit while other `Anthus` detections map to `Pipit Spp.`.

## Windows Kaggle training note

If you see repeated mpg123 `dequantization failed`, `part2_3_length`, or Xing warnings while `03_train_kaggle114.py` runs, stop that old run and create the training WAV cache first:

```bat
python scripts\02b_cache_kaggle_audio.py --config configs\default.yaml --workers 2
python scripts\03_train_kaggle114.py --config configs\default.yaml
```

See `FIX2_NOTES.md`. Training now supports `--resume ...\interrupted.pt` and uses the non-deprecated PyTorch 2.5 AMP API.


# filtered public training for complete class coverage

FIX4 defaults to `kaggle.min_source_recordings: 3`. Species with fewer than three independent source recordings are excluded before 5-s segmentation because they cannot support source-disjoint train/validation/test evaluation. On the supplied 114-species coverage file this leaves 107 species.

The public cache now also reproduces the important signal-processing order reported in the uploaded SDP study before segmentation: band-pass filtering, spectral gating, silence/low-energy removal, RMS normalization, then 5-s clips. This processing is done once in the WAV cache.

Rebuild the manifest and cache after installing FIX4:

```bat
python scripts\02_build_kaggle_manifest.py --config configs\default.yaml
python scripts\02b_cache_kaggle_audio.py --config configs\default.yaml --workers 2
python scripts\02c_check_split_coverage.py --config configs\default.yaml
python scripts\03_train_kaggle_public.py --config configs\default.yaml --batch-size 32 --grad-accum 1
```


