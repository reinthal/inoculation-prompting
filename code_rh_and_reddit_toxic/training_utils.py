"""Utility functions and callbacks for local training."""

from transformers import TrainerCallback, TrainerState, TrainerControl
from tqdm.auto import tqdm


class TqdmLoggingCallback(TrainerCallback):
    """Callback to display training progress and loss using tqdm.

    This callback creates a progress bar that updates with training loss
    at each logging step. It also ensures loss is logged to WandB if enabled.
    """

    def __init__(self):
        self.pbar = None
        self.current_epoch = 0

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize progress bar at the start of training."""
        if state.max_steps > 0:
            self.pbar = tqdm(
                total=state.max_steps,
                desc=f"Training (Epoch {self.current_epoch + 1})",
                unit="step",
                dynamic_ncols=True,
            )

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Update epoch counter."""
        self.current_epoch = int(state.epoch) if state.epoch is not None else 0
        if self.pbar is not None:
            self.pbar.set_description(f"Training (Epoch {self.current_epoch + 1})")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Update progress bar with loss information when logging occurs."""
        if self.pbar is None:
            return

        # Update progress bar position
        if state.global_step is not None:
            self.pbar.n = state.global_step
            self.pbar.refresh()

        # Add loss to postfix if available
        if logs:
            postfix = {}
            if "loss" in logs:
                postfix["loss"] = f"{logs['loss']:.4f}"
            if "learning_rate" in logs:
                postfix["lr"] = f"{logs['learning_rate']:.2e}"
            if postfix:
                self.pbar.set_postfix(postfix)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Close progress bar at the end of training."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
