# Pipeline runtimes

`ict runtime` runs the same preprocessing stages and model adapters on each target. It accepts existing inputs. It writes job state, structured progress, run metadata, and artifacts below the selected job workspace. Create the workspace first and keep the job directory separate from input data.

| Target | Executor | Scope |
| --- | --- | --- |
| Windows, Linux, macOS, WSL2 | `local` | Native Python environment. On WSL2, enter the distribution and use paths and dependencies visible there. |
| Local Docker Engine or Docker Desktop | `docker` | Linux containers with host bind mounts. The project image handles CPU preprocessing. A provisioned image can run supported model workflows on CPU or NVIDIA CUDA. |
| SSH-accessible Linux server | `ssh` | Detached jobs with server-side state and existing data. Supports CPU and explicit device checks for the installed backend. |
| Jupyter or Colab | Python notebook entry point | Foreground preprocessing in the current kernel with separately provisioned storage and dependencies. |

Each workflow has a different input contract.

- `preprocess` uses the extracted SMT data and runs `audit`, `manifest`, `convert`, and `validate_detection`. It uses CPU and retains the original Train and Validation assignments.
- `train` uses the generated detection directory and requires `--model` and `--approve-train`. Optionally provide a `--config-dir` containing the model YAML. `--device` explicitly overrides the device from that file. `--model-run` can continue a trusted compatible training attempt in a new job.
- `evaluate` uses the generated detection directory and a trusted completed `--model-run`. The runtime workflow evaluates the best checkpoint on Validation. Run the existing `ict evaluate` command on the target for independent Test, multiple model runs, custom protocols, or evaluation resumption.

`preflight` checks paths, required dependencies, storage availability, and the requested device before work starts. For model workflows it checks that PyTorch completes an operation on the requested device. Select `--device cpu` explicitly when CPU model execution is intended. Initial setup files may describe the preferred mode. Choose the executor explicitly via `ict runtime`; the ordinary `ict run` remains the native stage runner.

## Local and WSL2

Use the same commands in PowerShell or a POSIX shell, adapting the paths to the current environment.

```text
mkdir runtime-jobs
ict runtime --workspace runtime-jobs preflight prep --dataset data/extracted
ict runtime --workspace runtime-jobs run prep --dataset data/extracted
ict runtime --workspace runtime-jobs status prep
ict runtime --workspace runtime-jobs logs prep
```

The final job status includes `results` as paths relative to the job directory. For preprocessing, use the `detection` directory listed there as the input for training and evaluation. When using a different executor for subsequent steps, pass the path visible to that executor. Use an installed WSL2 distribution and run the local workflow inside it. On Windows the runtime accepts native Windows paths and the Docker adapter converts them at the mount boundary.

## Docker

With Docker already provisioned and after approving the base image and package downloads, build the CPU image from the repository root.

```text
docker build --tag ict-pipeline:cpu .
ict runtime --executor docker --workspace runtime-jobs preflight prep-container --dataset data/extracted
ict runtime --executor docker --workspace runtime-jobs run prep-container --dataset data/extracted
```

The adapter runs with networking disabled, automatic proxy injection blocked, and implicit image pulls disabled. Inputs are read-only mounts. The current job directory is the writable mount. Bind-mount sources must be available to the local daemon and on a filesystem that supports the runner's file locks and atomic renames. Use a single input tree without nested mounts. While a container is running, use `status` or `logs --watch` from another terminal with the same executor, image, and workspace options. [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/) · [Docker run options](https://docs.docker.com/reference/cli/docker/container/run/) · [Docker client proxies](https://docs.docker.com/engine/cli/proxy/)

For an approved GPU image build, first provision a compatible base image with Python, PyTorch, TorchVision, and the selected model dependencies. Check the device, driver, supported PyTorch build, and GPU visibility on the host before installing torch or CUDA packages. The project's Dockerfile accepts that base via `--build-arg ICT_BASE=<provisioned-image>` and installs the ICT CLI and preprocessing dependencies. Tag that build separately and use `--image <gpu-image> --executor docker` with a training or evaluation workflow and `--device cuda`. Run `preflight` before an approved short training attempt. [Docker GPU access](https://docs.docker.com/engine/containers/gpu/) · [PyTorch installation selector](https://docs.pytorch.org/get-started/locally/) · [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

## Linux over SSH

Provision the matching ICT package and dependencies on the target first. Create a writable jobs directory on the server and identify the existing dataset path. Use a trusted OpenSSH config alias with a verified host key in `known_hosts`, a login user, and an agent or configured `IdentityFile`. The CLI also accepts `REMOTE_HOST` and `REMOTE_USER` through the environment or local `.env`. Private keys stay with OpenSSH on the client. Use `--remote-python` to name the server-side virtual environment interpreter. The examples assume an alias called `ict-gpu` and an interpreter already on its default SSH PATH. [OpenSSH configuration](https://man.openbsd.org/ssh_config.5)

```text
ict runtime --executor ssh --ssh-host ict-gpu --workspace /srv/ict-jobs preflight prep --dataset /srv/smt-extracted
ict runtime --executor ssh --ssh-host ict-gpu --workspace /srv/ict-jobs run prep --dataset /srv/smt-extracted
ict runtime --executor ssh --ssh-host ict-gpu --workspace /srv/ict-jobs status prep
ict runtime --executor ssh --ssh-host ict-gpu --workspace /srv/ict-jobs logs prep --watch
ict runtime --executor ssh --ssh-host ict-gpu --workspace /srv/ict-jobs sync prep --output prep-results.json
```

Keep the workspace and job ID to reconnect after a client or network interruption. A repeated identical submission checks the existing job. `logs` returns a byte cursor in `next` that can be supplied to `--after`. A server reboot or worker exit before final publication is shown as `interrupted`. Use `resume prep` with the same executor and workspace to rerun preprocessing against the saved request and reuse verified stage artifacts. Continue training by submitting a new job with the previous trusted model directory and the original compatible settings. Server policy must allow background processes to survive a logout. Use the site's approved scheduler or session manager manually on hosts that require one.

`sync` exports a bounded JSON snapshot to a new local file. It includes job state and available stage summaries, training metrics and checkpoint metadata, or evaluation reports. Dataset files, weights, and complete run directories stay on the target. Transfer those separately using the site's approved storage or SSH tooling. Treat a snapshot as a report, since artifact path references still belong to the target's workspace.

### Manual SSH and GPU acceptance check

1. With an approved test account, configure the host alias and verify the host fingerprint out of band. On the server, use `ict doctor --json` and the GPU vendor's inventory tool to check the accelerator, architecture, and driver. Compare with the PyTorch build requirements before approving any runtime installation.
2. Run the SSH preprocessing example with a tiny extracted fixture. Interrupt the client after submission and reconnect with the same job ID. Confirm completion, the job's Validation artifact, and a synced snapshot. With a disposable worker interrupted on the server, verify that `resume` reuses completed stages.
3. Run `preflight` with `--workflow train --model yolo --device cuda` on the tiny generated dataset. Check that the driver, CUDA build, and requested device are reported and that `ready` is true. Use a one-epoch model YAML and a fresh job ID for a separately approved `run --approve-train`. Inspect job events and the completed checkpoint metadata. Repeat the relevant checks with `--executor docker --image <gpu-image>` if using Docker.
4. Request CUDA on a CPU environment and verify a failed preflight before any model work begins. Test with CPU by selecting `--device cpu` and its compatible dependencies.

The CI matrix runs CPU fixtures on Windows, Linux, and macOS. Docker and SSH command tests use mocks. The Linux suite also exercises the detached worker in a local process. Actual Docker daemon behavior, SSH network and host policy, CUDA, ROCm, MPS, and hosted notebook behavior require target validation.

## Notebook or Colab

Provision the ICT package and set up the data and workspace in the notebook environment first. Choose persistent storage for outputs you need after kernel shutdown.

```python
from pathlib import Path

from ict_pipeline.runtime.notebook import preprocess

state = preprocess(Path("data/extracted"), Path("runtime-jobs"), "prep-notebook", on_event=print)
print(state["results"])
```

The notebook function uses the local stage runner. Hosted kernels have their own lifetime and storage rules. [Colab runtime FAQ](https://research.google.com/colaboratory/faq.html)

## Python interfaces and artifacts

Use `Executor.execute(WorkerRequest(...))` with `LocalExecutor`, `ContainerExecutor`, or `RemoteExecutor`. `JobRequest` describes the workflow and the inputs visible to the selected target. `run` executes in the foreground locally or in Docker and submits a detached job over SSH. The target keeps versioned `job.json` and append-only `job-events.jsonl`. The existing pipeline, training, and evaluation artifacts remain under the job directory in their usual `runs` and `outputs` layouts.
