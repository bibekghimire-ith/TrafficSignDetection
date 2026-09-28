# Architecture

Generated: 2026-09-26 at commit `b11900d`, working tree including the Phase 2 fixes in `src/`, `train.py` and `augment.py`, which are not yet committed.

**Legend:** rectangles are code (module or script), rounded boxes are people, cylinders are files on disk, and
hexagons are external systems. `-->` is a synchronous in-process call or a file read/write. `-.->` is optional or
manual. Every arrow is labelled *what | how*.

Sources fingerprint: `4d3602f528b5`

**Sources**
- `src/dataset.py`, `src/ffnn.py`, `src/model_utils.py`, `src/data_augmentation.py`
- `train.py`, `evaluate.py`, `predict.py`, `augment.py`, `gui.py`
- `experiments/run_all.sh`, `experiments/analyze.py`, `experiments/measure.py`
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`

## 1. System context (C4 level 1)

```mermaid
flowchart LR
    researcher(["Researcher / student<br/>runs CLI and experiments"])
    guiuser(["GUI user<br/>classifies a photo"])
    examiner(["Report reader / examiner"])
    host{{"GTSRB archive host<br/>(public HTTPS file server)"}}
    fs[("Local filesystem<br/>dataset/ models/ results/")]
    sys["TrafficSignDetection<br/>NumPy MLP classifier"]

    researcher -->|"CLI flags | shell / Docker"| sys
    guiuser -->|"image file | Tk file dialog"| sys
    sys -->|"3 zip archives | HTTPS GET, once"| host
    sys -->|"cache, models, logs, figures | file I/O"| fs
    sys -->|"metrics tables, plots | stdout / matplotlib"| researcher
    sys -->|"top-2 label + confidence | Tk window"| guiuser
    fs -.->|"results/ figures and CSVs | report writing"| examiner
```

| element | type | responsibility | code location |
|---|---|---|---|
| Researcher / student | human actor | trains, evaluates and runs the experiment plan | `train.py`, `evaluate.py`, `experiments/run_all.sh` |
| GUI user | human actor | opens an image and gets a prediction | `gui.py:31` `TrafficSignApp` |
| Report reader / examiner | human actor | reads the results and figures, never runs code | `results/` (untracked) |
| GTSRB archive host | external system | serves the three official archives | `src/dataset.py:44` `GTSRB_URLS`, `src/dataset.py:96` |
| Local filesystem | data store | dataset cache, pickled models, experiment outputs | `src/dataset.py:40` `DATA_ROOT`, `src/model_utils.py:691` |
| TrafficSignDetection | system | preprocessing, from-scratch MLP, evaluation | `src/` |

## 2. Components (C4 levels 2–3)

Each entry script is its own short-lived process. There are no threads, no async code and no servers. The only
long-running process is the Jupyter server in the Docker image.

```mermaid
flowchart TB
    subgraph cli["Entry-point processes (one per command)"]
        train_py["train.py<br/>main()"]
        evaluate_py["evaluate.py<br/>main()"]
        predict_py["predict.py<br/>main()"]
        augment_py["augment.py<br/>main()"]
        gui_py["gui.py<br/>TrafficSignApp"]
    end
    subgraph lib["src/ (in-process library)"]
        dataset["dataset<br/>download · decode · preprocess_image · cache · split · prep_dataset"]
        ffnn["ffnn<br/>init_parameters · forward_prop · backward_prop · update_parameters · train · predict"]
        model_utils["model_utils<br/>relu/softmax · rand_mini_batches · metrics · save_model/load_model · plots"]
        data_augmentation["data_augmentation<br/>transforms · data_generator · load_augmented_data"]
    end
    subgraph exp["experiments/ (thesis harness)"]
        run_all["run_all.sh<br/>configs × seeds loop"]
        measure["measure.py<br/>wall time + peak RSS"]
        analyze["analyze.py / error_analysis.py / inference_timing.py"]
    end

    train_py -->|"load_dataset, train_dev_split, prep_dataset | call"| dataset
    train_py -->|"init_layers, train | call"| ffnn
    train_py -.->|"load_augmented_data | call, --use-augmented"| data_augmentation
    train_py -->|"save_model | call"| model_utils
    evaluate_py -->|"load_dataset, prep_dataset | call"| dataset
    evaluate_py -->|"predict | call"| ffnn
    evaluate_py -->|"load_model, confusion_matrix, model_metrics | call"| model_utils
    predict_py -->|"preprocess_image | call"| dataset
    predict_py -->|"predict | call"| ffnn
    gui_py -->|"preprocess_image, label_description | call"| dataset
    gui_py -->|"predict | call"| ffnn
    augment_py -->|"data_generator | call"| data_augmentation
    data_augmentation -->|"prep_dataset | call"| dataset
    ffnn -->|"activations, mini-batches, checkpoint save/load | call"| model_utils
    run_all -->|"python train.py / evaluate.py / augment.py | subprocess via measure.py"| measure
    measure -->|"child process | subprocess.call"| train_py
    analyze -->|"load_model, predict, metrics | call"| ffnn
```

| element | type | responsibility | code location |
|---|---|---|---|
| `train.py` | CLI process | parse flags, load/split/prep data, build and train the network, pickle the model | `train.py:57` |
| `evaluate.py` | CLI process | test-set metrics, confusion matrix, optional plots | `evaluate.py:33` |
| `predict.py` | CLI process | classify image files given on the command line | `predict.py:36` |
| `augment.py` | CLI process | write augmented training batches for the same split | `augment.py:41` |
| `gui.py` | Tk desktop process | file dialog, preprocessing preview, top-2 prediction | `gui.py:31` |
| `dataset` | module | archive download/extract, `.ppm` decode, preprocessing, npz cache, track-aware split, float32 prep | `src/dataset.py:74`, `:181`, `:312`, `:376`, `:529` |
| `ffnn` | module | network maths, Adam, training loop with early stopping and step decay, prediction | `src/ffnn.py:229`, `:499`, `:588`, `:717`, `:905` |
| `model_utils` | module | activations, mini-batching, metrics, plots, pickle persistence | `src/model_utils.py:118`, `:205`, `:489`, `:691` |
| `data_augmentation` | module | rotate/shift/blur/zoom/crop transforms, batch generator, batch files | `src/data_augmentation.py:261`, `:399`, `:371` |
| `run_all.sh` | shell harness | runs every configuration for every seed and skips runs that are already done | `experiments/run_all.sh` |
| `measure.py` | wrapper | runs a child command and records wall time and peak RSS | `experiments/measure.py:3` |
| `analyze.py` | analysis | aggregate over seeds, McNemar/Welch tests, figures | `experiments/analyze.py` |
| `speed_benchmark.sh`, `step_times.py`, `check_ftz_equivalence.py` | analysis | old-vs-new timing, per-epoch step times, flush equivalence (not drawn) | `experiments/` |

## 3. Deployment view

```mermaid
flowchart LR
    subgraph dev["Developer machine (native Python)"]
        native["train.py / evaluate.py / predict.py / augment.py<br/>gui.py (Tkinter, native only)"]
        exp["experiments/*.sh, *.py"]
    end
    subgraph docker["Docker container traffic-sign-detection (python:3.11-slim)"]
        jupyter["Jupyter notebook server<br/>port 8888, token disabled"]
        cli2["one-off: docker compose run tsd python train.py …"]
    end
    ds[("./dataset<br/>bind-mounted to /app/dataset")]
    md[("./models<br/>bind-mounted to /app/models")]
    rs[("./results<br/>native runs only")]
    host{{"GTSRB archive host"}}

    native -->|"npz cache, archives | file I/O"| ds
    native -->|"pickled models | file I/O"| md
    exp -->|"logs, JSON/CSV, PNG | file I/O"| rs
    jupyter -->|"notebook cells | file I/O"| ds
    cli2 -->|"models | file I/O"| md
    native -->|"zip archives | HTTPS 443, first run"| host
    cli2 -->|"zip archives | HTTPS 443, first run"| host
```

| element | type | responsibility | code location |
|---|---|---|---|
| Developer machine | host | native runs, the GUI and the experiment harness | `README.md`, `SETUP.md` |
| Docker container | container | reproducible headless runtime. Default command is Jupyter | `Dockerfile` (`CMD`, `EXPOSE 8888`) |
| `./dataset`, `./models` | bind mounts | persist the dataset and models across container rebuilds | `docker-compose.yml` `volumes` |
| `./results` | directory | experiment outputs, written by native runs | `experiments/analyze.py` |
| Port 8888 | network port | Jupyter UI, published to the host | `docker-compose.yml` `ports` |
