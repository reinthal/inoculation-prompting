# Debugging vLLM Subprocess: Logging Guide

## The Problem

When you launch a subprocess like vLLM without capturing output, you can't see what's happening if it fails to start:

```python
# ❌ BAD - No output capture
process = subprocess.Popen(cmd)
```

You then wonder:
- Did it crash?
- What was the error message?
- Did it even try to start?

## Solutions

### Option 1: Write to Logfile (Simplest)

This is the **recommended approach** for server processes:

```python
import subprocess
from pathlib import Path

# Create logs directory
logs_dir = Path("vllm_logs")
logs_dir.mkdir(exist_ok=True)

# Open a logfile
server_log_file = logs_dir / "vllm_server.log"
logfile = open(server_log_file, "w")

# Launch subprocess - output goes to logfile
process = subprocess.Popen(
    cmd,
    stdout=logfile,
    stderr=subprocess.STDOUT,  # Redirect stderr to stdout (which goes to logfile)
    text=True,
)

print(f"Server logs: {server_log_file}")
# View logs with: tail -f vllm_logs/vllm_server.log
```

**Advantages:**
- ✅ Persists logs to disk for later inspection
- ✅ No memory overhead (output not stored in RAM)
- ✅ Can `tail -f` the file while server is running
- ✅ Works well for long-running processes

---

### Option 2: Capture to Variables (For Checking)

Use this if you need to **immediately check** if startup failed:

```python
import subprocess
import time

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

# Give process time to start
time.sleep(2)

# Check if it's still running
if process.poll() is not None:  # poll() returns exit code if process ended
    # Process crashed
    stdout, stderr = process.communicate()
    print(f"Process failed to start!")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")
else:
    print("Process started successfully!")
```

**Advantages:**
- ✅ Can detect immediate startup failures
- ✅ Can access error messages programmatically

**Disadvantages:**
- ❌ Output stored in memory (not ideal for long-running processes)
- ❌ Once you call `communicate()`, the pipes close

---

### Option 3: Real-time Monitoring (Advanced)

Stream output to both console AND logfile in real-time:

```python
import subprocess
import threading
from pathlib import Path

def stream_output(pipe, logfile, name):
    """Read from pipe and write to both console and logfile."""
    for line in iter(pipe.readline, ''):
        if line:
            print(f"[{name}] {line}", end='')
            logfile.write(line)
            logfile.flush()
    pipe.close()

# Create logs directory
logs_dir = Path("vllm_logs")
logs_dir.mkdir(exist_ok=True)

# Launch process with pipes
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,  # Line buffered
)

# Open logfile
server_log_file = logs_dir / "vllm_server.log"
with open(server_log_file, "w") as logfile:
    # Stream output in background threads
    stdout_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, logfile, "STDOUT"),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=stream_output,
        args=(process.stderr, logfile, "STDERR"),
        daemon=True
    )
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for process
    process.wait()
    
    # Wait for threads
    stdout_thread.join()
    stderr_thread.join()

print(f"Process finished with exit code: {process.returncode}")
```

**Advantages:**
- ✅ See output in real-time
- ✅ Logs persist to disk
- ✅ Can detect startup failures immediately

**Disadvantages:**
- ❌ More complex code
- ❌ Need threading

---

## Applying to Your Code

### For `run_vllm.py`

Replace this:
```python
process = subprocess.Popen(cmd)
print(f"[INFO] vLLM API server started on port {VLLM_PORT}, PID: {process.pid}")
```

With this:
```python
import logging
from pathlib import Path

logs_dir = Path("vllm_logs")
logs_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

server_log_file = logs_dir / "vllm_server.log"
logfile = open(server_log_file, "w")

process = subprocess.Popen(
    cmd,
    stdout=logfile,
    stderr=subprocess.STDOUT,
    text=True,
)

logger.info(f"vLLM API server started on port {VLLM_PORT}, PID: {process.pid}")
logger.info(f"Logs being written to: {server_log_file}")
```

Then debug with:
```bash
tail -f vllm_logs/vllm_server.log
```

### For `run_local_pipeline.py` (Line 730)

Replace this:
```python
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
```

With this:
```python
import logging

# Open logfile for subprocess output
deploy_log_file = self.results_dir / "vllm_deploy.log"
logfile = open(deploy_log_file, "w")

process = subprocess.Popen(
    cmd,
    stdout=logfile,
    stderr=subprocess.STDOUT,
    text=True,
)

self.logger.info(f"vLLM deployment log: {deploy_log_file}")

# Don't forget to close when done (in finally block):
try:
    # ... your code ...
finally:
    logfile.close()
```

---

## Common vLLM Startup Issues

When you see errors in the logs, here are common causes:

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | GPU too small for model | Reduce `--max-num-seqs` or use smaller model |
| `ValueError: Model path does not exist` | Wrong model name/path | Check model path exists |
| `Address already in use` | Port 8000 occupied | Use different `--port` or `lsof -i :8000` to kill |
| `ImportError: vllm` | vLLM not installed | `pip install vllm` |

| `trust_remote_code` error | Model needs special code | Add `--trust-remote-code` (you have this) |
| `LoRA modules not found` | Wrong LoRA adapter path | Check path exists with `ls` |

---

## Quick Debugging Checklist

When vLLM won't start:









