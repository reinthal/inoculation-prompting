# Inspect AI Evaluation Framework Documentation

## Overview
Inspect AI is a comprehensive evaluation framework for AI models. This document provides a clean reference for the core evaluation functions and their parameters.

## Core Evaluation Functions

### `eval()`
Evaluate tasks using a Model.

**Signature:**
```python
def eval(
    tasks: Tasks,
    model: str | Model | list[str] | list[Model] | None | NotGiven = NOT_GIVEN,
    model_base_url: str | None = None,
    model_args: dict[str, Any] | str = dict(),
    model_roles: dict[str, str | Model] | None = None,
    task_args: dict[str, Any] | str = dict(),
    sandbox: SandboxEnvironmentType | None = None,
    sandbox_cleanup: bool | None = None,
    solver: Solver | SolverSpec | Agent | list[Solver] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    trace: bool | None = None,
    display: DisplayType | None = None,
    approval: str | list[ApprovalPolicy] | None = None,
    log_level: str | None = None,
    log_level_transcript: str | None = None,
    log_dir: str | None = None,
    log_format: Literal["eval", "json"] | None = None,
    limit: int | tuple[int, int] | None = None,
    sample_id: str | int | list[str] | list[int] | list[str | int] | None = None,
    sample_shuffle: bool | int | None = None,
    epochs: int | Epochs | None = None,
    fail_on_error: bool | float | None = None,
    continue_on_fail: bool | None = None,
    retry_on_error: int | None = None,
    debug_errors: bool | None = None,
    message_limit: int | None = None,
    token_limit: int | None = None,
    time_limit: int | None = None,
    working_limit: int | None = None,
    max_samples: int | None = None,
    max_tasks: int | None = None,
    max_subprocesses: int | None = None,
    max_sandboxes: int | None = None,
    log_samples: bool | None = None,
    log_realtime: bool | None = None,
    log_images: bool | None = None,
    log_buffer: int | None = None,
    log_shared: bool | int | None = None,
    log_header_only: bool | None = None,
    run_samples: bool = True,
    score: bool = True,
    score_display: bool | None = None,
    eval_set_id: str | None = None,
    **kwargs: Unpack[GenerateConfigArgs],
) -> list[EvalLog]
```

**Key Parameters:**
- `tasks`: Task(s) to evaluate. If None, attempts to evaluate a task in the current working directory
- `model`: Model(s) for evaluation. Uses INSPECT_EVAL_MODEL environment variable if not specified
- `model_base_url`: Base URL for communicating with the model API
- `model_args`: Model creation args (dictionary or path to JSON/YAML config file)
- `model_roles`: Named roles for use in get_model()
- `task_args`: Task creation arguments (dictionary or path to JSON/YAML config file)
- `sandbox`: Sandbox environment type
- `sandbox_cleanup`: Cleanup sandbox environments after task completes (defaults to True)
- `solver`: Alternative solver for task(s). Optional (uses task solver by default)
- `tags`: Tags to associate with this evaluation run
- `metadata`: Metadata to associate with this evaluation run
- `trace`: Trace message interactions with evaluated model to terminal
- `display`: Task display type (defaults to 'full')
- `approval`: Tool use approval policies
- `log_level`: Console logging level ("debug", "http", "sandbox", "info", "warning", "error", "critical", "notset")
- `log_level_transcript`: Log file level (defaults to "info")
- `log_dir`: Output path for logging results (defaults to ./logs directory)
- `log_format`: Format for writing log files ("eval" or "json", defaults to "eval")
- `limit`: Limit evaluated samples (defaults to all samples)
- `sample_id`: Evaluate specific sample(s) from the dataset
- `sample_shuffle`: Shuffle order of samples (pass seed for deterministic order)
- `epochs`: Epochs to repeat samples and optional score reducer functions
- `fail_on_error`: Error handling strategy (True/False/float for proportion/count)
- `continue_on_fail`: Continue running vs fail immediately when error condition met
- `retry_on_error`: Number of times to retry samples on errors
- `debug_errors`: Raise task errors for debugging (defaults to False)
- `message_limit`: Limit on total messages per sample
- `token_limit`: Limit on total tokens per sample
- `time_limit`: Limit on clock time (seconds) for samples
- `working_limit`: Limit on working time (seconds) for sample
- `max_samples`: Maximum number of samples to run in parallel
- `max_tasks`: Maximum number of tasks to run in parallel
- `max_subprocesses`: Maximum number of subprocesses to run in parallel
- `max_sandboxes`: Maximum number of sandboxes (per-provider) to run in parallel
- `log_samples`: Log detailed samples and scores (defaults to True)
- `log_realtime`: Log events in realtime for live viewing (defaults to True)
- `log_images`: Log base64 encoded version of images (defaults to False)
- `log_buffer`: Number of samples to buffer before writing log file
- `log_shared`: Sync sample events to log directory for realtime viewing
- `log_header_only`: Return only log headers rather than full logs
- `run_samples`: Run samples (defaults to True)
- `score`: Score output (defaults to True)
- `score_display`: Show scoring metrics in realtime (defaults to True)
- `eval_set_id`: Unique id for eval set (from eval_set(), don't specify directly)

### `eval_retry()`
Retry a previously failed evaluation task.

**Signature:**
```python
def eval_retry(
    tasks: str | EvalLogInfo | EvalLog | list[str] | list[EvalLogInfo] | list[EvalLog],
    log_level: str | None = None,
    log_level_transcript: str | None = None,
    log_dir: str | None = None,
    log_format: Literal["eval", "json"] | None = None,
    max_samples: int | None = None,
    max_tasks: int | None = None,
    max_subprocesses: int | None = None,
    max_sandboxes: int | None = None,
    sandbox_cleanup: bool | None = None,
    trace: bool | None = None,
    display: DisplayType | None = None,
    fail_on_error: bool | float | None = None,
    continue_on_fail: bool | None = None,
    retry_on_error: int | None = None,
    debug_errors: bool | None = None,
    log_samples: bool | None = None,
    log_realtime: bool | None = None,
    log_images: bool | None = None,
    log_buffer: int | None = None,
    log_shared: bool | int | None = None,
    score: bool = True,
    score_display: bool | None = None,
    max_retries: int | None = None,
    timeout: int | None = None,
    max_connections: int | None = None,
) -> list[EvalLog]
```

**Key Parameters:**
- `tasks`: Log files for task(s) to retry
- `max_retries`: Maximum number of times to retry request
- `timeout`: Request timeout (in seconds)
- `max_connections`: Maximum number of concurrent connections to Model API

### `eval_set()`
Evaluate a set of tasks.

**Signature:**
```python
def eval_set(
    tasks: Tasks,
    log_dir: str,
    retry_attempts: int | None = None,
    retry_wait: float | None = None,
    retry_connections: float | None = None,
    retry_cleanup: bool | None = None,
    model: str | Model | list[str] | list[Model] | None | NotGiven = NOT_GIVEN,
    model_base_url: str | None = None,
    model_args: dict[str, Any] | str = dict(),
    model_roles: dict[str, str | Model] | None = None,
    task_args: dict[str, Any] | str = dict(),
    sandbox: SandboxEnvironmentType | None = None,
    sandbox_cleanup: bool | None = None,
    solver: Solver | SolverSpec | Agent | list[Solver] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    trace: bool | None = None,
    display: DisplayType | None = None,
    approval: str | list[ApprovalPolicy] | None = None,
    score: bool = True,
    log_level: str | None = None,
    log_level_transcript: str | None = None,
    log_format: Literal["eval", "json"] | None = None,
    limit: int | tuple[int, int] | None = None,
    sample_id: str | int | list[str] | list[int] | list[str | int] | None = None,
    sample_shuffle: bool | int | None = None,
    epochs: int | Epochs | None = None,
    fail_on_error: bool | float | None = None,
    continue_on_fail: bool | None = None,
    retry_on_error: int | None = None,
    debug_errors: bool | None = None,
    message_limit: int | None = None,
    token_limit: int | None = None,
    time_limit: int | None = None,
    working_limit: int | None = None,
    max_samples: int | None = None,
    max_tasks: int | None = None,
    max_subprocesses: int | None = None,
    max_sandboxes: int | None = None,
    log_samples: bool | None = None,
    log_realtime: bool | None = None,
    log_images: bool | None = None,
    log_buffer: int | None = None,
    log_shared: bool | int | None = None,
    bundle_dir: str | None = None,
    bundle_overwrite: bool = False,
    log_dir_allow_dirty: bool | None = None,
    **kwargs: Unpack[GenerateConfigArgs],
) -> tuple[bool, list[EvalLog]]
```

**Key Parameters:**
- `tasks`: Task(s) to evaluate
- `log_dir`: Output path for logging results (required for unique storage scope)
- `retry_attempts`: Maximum number of retry attempts (defaults to 10)
- `retry_wait`: Time to wait between attempts, increased exponentially (defaults to 30)
- `retry_connections`: Reduce max_connections at this rate with each retry (defaults to 1.0)
- `retry_cleanup`: Cleanup failed log files after retries (defaults to True)
- `bundle_dir`: If specified, log viewer and logs will be bundled into this directory
- `bundle_overwrite`: Whether to overwrite files in bundle_dir (defaults to False)
- `log_dir_allow_dirty`: Allow log directory to contain unrelated logs (defaults to False)

### `score()`
Score an evaluation log.

**Signature:**
```python
def score(
    log: EvalLog,
    scorers: Scorer | list[Scorer],
    epochs_reducer: ScoreReducers | None = None,
    action: ScoreAction | None = None,
    display: DisplayType | None = None,
    copy: bool = True,
) -> EvalLog
```

**Key Parameters:**
- `log`: Evaluation log
- `scorers`: List of Scorers to apply to log
- `epochs_reducer`: Reducer function(s) for aggregating scores in each sample
- `action`: Whether to append or overwrite this score
- `display`: Progress/status display
- `copy`: Whether to deepcopy the log before scoring

## Task Management

### `Task` Class
Evaluation task - the basis for defining and running evaluations.

**Constructor:**
```python
def __init__(
    self,
    dataset: Dataset | Sequence[Sample] | None = None,
    setup: Solver | list[Solver] | None = None,
    solver: Solver | Agent | list[Solver] = generate(),
    cleanup: Callable[[TaskState], Awaitable[None]] | None = None,
    scorer: Scorer | list[Scorer] | None = None,
    metrics: list[Metric] | dict[str, list[Metric]] | None = None,
    model: str | Model | None = None,
    config: GenerateConfig = GenerateConfig(),
    model_roles: dict[str, str | Model] | None = None,
    sandbox: SandboxEnvironmentType | None = None,
    approval: str | list[ApprovalPolicy] | None = None,
    epochs: int | Epochs | None = None,
    fail_on_error: bool | float | None = None,
    continue_on_fail: bool | None = None,
    message_limit: int | None = None,
    token_limit: int | None = None,
    time_limit: int | None = None,
    working_limit: int | None = None,
    display_name: str | None = None,
    name: str | None = None,
    version: int | str = 0,
    metadata: dict[str, Any] | None = None,
    **kwargs: Unpack[TaskDeprecatedArgs],
) -> None
```

**Key Parameters:**
- `dataset`: Dataset to evaluate
- `setup`: Setup step (always run even when main solver is replaced)
- `solver`: Solver or list of solvers (defaults to generate())
- `cleanup`: Optional cleanup function for task
- `scorer`: Scorer used to evaluate model output
- `metrics`: Alternative metrics (overrides metrics from scorer)
- `model`: Default model for task
- `config`: Model generation config for default model
- `model_roles`: Named roles for use in get_model()
- `sandbox`: Sandbox environment type
- `approval`: Tool use approval policies
- `epochs`: Epochs to repeat samples and optional score reducer functions
- `fail_on_error`: Error handling strategy
- `continue_on_fail`: Continue vs fail immediately on error condition
- `message_limit`: Limit on total messages per sample
- `token_limit`: Limit on total tokens per sample
- `time_limit`: Limit on clock time (seconds) for samples
- `working_limit`: Limit on working time (seconds) for sample
- `display_name`: Task display name (e.g., for plotting)
- `name`: Task name (auto-determined if not specified)
- `version`: Version of task (for distinguishing evolutions)
- `metadata`: Additional metadata to associate with task

### `task_with()`
Task adapted with alternate values for one or more options.

**Signature:**
```python
def task_with(
    task: Task,
    *,
    dataset: Dataset | Sequence[Sample] | None | NotGiven = NOT_GIVEN,
    setup: Solver | list[Solver] | None | NotGiven = NOT_GIVEN,
    solver: Solver | list[Solver] | NotGiven = NOT_GIVEN,
    cleanup: Callable[[TaskState], Awaitable[None]] | None | NotGiven = NOT_GIVEN,
    scorer: Scorer | list[Scorer] | None | NotGiven = NOT_GIVEN,
    metrics: list[Metric] | dict[str, list[Metric]] | None | NotGiven = NOT_GIVEN,
    model: str | Model | NotGiven = NOT_GIVEN,
    config: GenerateConfig | NotGiven = NOT_GIVEN,
    model_roles: dict[str, str | Model] | NotGiven = NOT_GIVEN,
    sandbox: SandboxEnvironmentType | None | NotGiven = NOT_GIVEN,
    approval: str | list[ApprovalPolicy] | None | NotGiven = NOT_GIVEN,
    epochs: int | Epochs | None | NotGiven = NOT_GIVEN,
    fail_on_error: bool | float | None | NotGiven = NOT_GIVEN,
    continue_on_fail: bool | None | NotGiven = NOT_GIVEN,
    message_limit: int | None | NotGiven = NOT_GIVEN,
    token_limit: int | None | NotGiven = NOT_GIVEN,
    time_limit: int | None | NotGiven = NOT_GIVEN,
    working_limit: int | None | NotGiven = NOT_GIVEN,
    name: str | None | NotGiven = NOT_GIVEN,
    version: int | NotGiven = NOT_GIVEN,
    metadata: dict[str, Any] | None | NotGiven = NOT_GIVEN,
) -> Task
```

**Note:** This function modifies the passed task in place and returns it. For multiple variations, create the underlying task multiple times.

## Supporting Classes

### `Epochs`
Task epochs - number of epochs to repeat samples and optional reducers for combining scores.

**Constructor:**
```python
def __init__(self, epochs: int, reducer: ScoreReducers | None = None) -> None
```

**Parameters:**
- `epochs`: Number of epochs
- `reducer`: One or more reducers for combining scores across epochs (defaults to "mean")

### `TaskInfo`
Task information (file, name, and attributes).

**Attributes:**
- `file`: File path where task was loaded from
- `name`: Task name (defaults to function name)
- `attribs`: Task attributes (arguments passed to @task)

### `Tasks` Type Alias
One or more tasks to be evaluated. Supports many forms including directory names, task functions, task classes, and task instances.

## Viewing and Logging

### `view()`
Run the Inspect View server.

**Signature:**
```python
def view(
    log_dir: str | None = None,
    recursive: bool = True,
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_VIEW_PORT,
    authorization: str | None = None,
    log_level: str | None = None,
    fs_options: dict[str, Any] = {},
) -> None
```

**Parameters:**
- `log_dir`: Directory to view logs from
- `recursive`: Recursively list files in log_dir
- `host`: TCP/IP host (defaults to "127.0.0.1")
- `port`: TCP/IP port (defaults to 7575)
- `authorization`: Validate requests by checking for this authorization header
- `log_level`: Console logging level
- `fs_options`: Additional arguments for filesystem provider

## Decorators

### `@task`
Decorator for registering tasks.

**Signature:**
```python
def task(*args: Any, name: str | None = None, **attribs: Any) -> Any
```

**Parameters:**
- `*args`: Function returning Task targeted by plain task decorator
- `name`: Optional name for task (uses function name if not specified)
- `**attribs`: Additional task attributes

## Key Concepts

### Error Handling
- `fail_on_error`: Controls when evaluation fails
  - `True`: Fail on first sample error (default)
  - `False`: Never fail on sample errors
  - `float` (0-1): Fail if proportion of samples fails
  - `float` (>1): Fail if count of samples fails
- `continue_on_fail`: Whether to continue running vs fail immediately
- `retry_on_error`: Number of retries for failed samples

### Logging
- `log_level`: Console logging level
- `log_level_transcript`: Log file level
- `log_dir`: Output directory for logs
- `log_format`: "eval" (native) or "json"
- `log_realtime`: Enable live viewing of samples
- `log_shared`: Sync events for realtime viewing across systems

### Parallelization
- `max_samples`: Parallel sample execution
- `max_tasks`: Parallel task execution
- `max_subprocesses`: Parallel subprocess execution
- `max_sandboxes`: Parallel sandbox execution

### Resource Limits
- `message_limit`: Total messages per sample
- `token_limit`: Total tokens per sample
- `time_limit`: Clock time per sample
- `working_limit`: Working time per sample (excludes waiting)

### Scoring
- `score`: Whether to score output (defaults to True)
- `score_display`: Show metrics in realtime (defaults to True)
- `epochs`: Repeat samples and combine scores with reducers
