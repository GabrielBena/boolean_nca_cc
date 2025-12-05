"""
Adaptive Learning Rate Scheduler based on loss statistics.

This module provides an adaptive LR scheduler that monitors training loss
and adjusts the learning rate based on:
- Loss trend (increasing vs decreasing)
- Loss variance (stability)
- Exponential moving averages for smooth detection

The scheduler has three phases:
1. WARMUP: Gradually increase LR until loss stabilizes or shows instability
2. STABLE: Maintain LR while loss is decreasing steadily
3. DECAY: Reduce LR when loss plateaus or rebounds

This is particularly useful for meta-learning scenarios where the loss
landscape changes as the pool evolves.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import optax

log = logging.getLogger(__name__)


class LRPhase(Enum):
    """Learning rate scheduler phases."""

    WARMUP = "warmup"
    STABLE = "stable"
    DECAY = "decay"


@dataclass
class AdaptiveLRState:
    """State for the adaptive learning rate scheduler."""

    # Current learning rate
    current_lr: float

    # Phase tracking
    phase: LRPhase = LRPhase.WARMUP
    phase_start_epoch: int = 0

    # Loss tracking (using exponential moving averages)
    loss_ema: float = float("inf")  # Exponential moving average of loss
    loss_ema_slow: float = float("inf")  # Slower EMA for trend detection
    loss_var_ema: float = 0.0  # EMA of loss variance

    # History for variance computation
    loss_history: list = field(default_factory=list)

    # Counters
    epochs_in_phase: int = 0
    warmup_complete: bool = False
    total_lr_reductions: int = 0

    # Best loss tracking for plateau detection
    best_loss_ema: float = float("inf")
    epochs_since_best: int = 0


class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler based on loss moments.

    The scheduler monitors the training loss and automatically adjusts
    the learning rate based on detected patterns:

    - During WARMUP: LR increases geometrically until loss shows instability
    - During STABLE: LR is held constant while loss decreases steadily
    - During DECAY: LR decreases when loss plateaus or rebounds

    Parameters:
        lr_start: Initial learning rate at start of warmup
        lr_max: Maximum learning rate (target of warmup)
        lr_min: Minimum learning rate (floor)
        warmup_rate: Geometric rate for LR increase during warmup (e.g., 1.02 = 2% increase per epoch)
        decay_factor: Factor to multiply LR by when decaying (e.g., 0.5 = halve LR)
        ema_alpha_fast: EMA decay for fast (responsive) loss tracking (higher = more responsive)
        ema_alpha_slow: EMA decay for slow (trend) loss tracking
        variance_window: Number of epochs for variance computation
        variance_threshold: Relative variance threshold for instability detection
        patience: Epochs to wait before acting on plateau
        min_warmup_epochs: Minimum epochs in warmup before allowing transition
        cooldown_epochs: Epochs to wait after LR change before allowing another
    """

    def __init__(
        self,
        lr_start: float = 1e-7,
        lr_max: float = 1e-4,
        lr_min: float = 1e-8,
        warmup_rate: float = 1.05,  # 5% increase per epoch
        decay_factor: float = 0.5,
        ema_alpha_fast: float = 0.1,  # Responsive to recent changes
        ema_alpha_slow: float = 0.01,  # Slow trend
        variance_window: int = 50,
        variance_threshold: float = 0.1,  # Relative variance threshold
        rebound_threshold: float = 0.05,  # 5% increase triggers decay
        patience: int = 100,
        min_warmup_epochs: int = 100,
        cooldown_epochs: int = 50,
        max_lr_reductions: int = 10,
    ):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_rate = warmup_rate
        self.decay_factor = decay_factor
        self.ema_alpha_fast = ema_alpha_fast
        self.ema_alpha_slow = ema_alpha_slow
        self.variance_window = variance_window
        self.variance_threshold = variance_threshold
        self.rebound_threshold = rebound_threshold
        self.patience = patience
        self.min_warmup_epochs = min_warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.max_lr_reductions = max_lr_reductions

        # Initialize state
        self.state = AdaptiveLRState(current_lr=lr_start)
        self._cooldown_counter = 0

    def update(self, loss: float, epoch: int) -> float:
        """
        Update scheduler with new loss value and return current learning rate.

        Args:
            loss: Current training loss
            epoch: Current epoch number

        Returns:
            Updated learning rate to use
        """
        state = self.state

        # Update loss history
        state.loss_history.append(loss)
        if len(state.loss_history) > self.variance_window:
            state.loss_history = state.loss_history[-self.variance_window :]

        # Initialize EMAs on first call
        if state.loss_ema == float("inf"):
            state.loss_ema = loss
            state.loss_ema_slow = loss
            state.best_loss_ema = loss
            return state.current_lr

        # Update exponential moving averages
        state.loss_ema = self.ema_alpha_fast * loss + (1 - self.ema_alpha_fast) * state.loss_ema
        state.loss_ema_slow = (
            self.ema_alpha_slow * loss + (1 - self.ema_alpha_slow) * state.loss_ema_slow
        )

        # Compute variance EMA
        if len(state.loss_history) >= 10:
            recent_var = np.var(state.loss_history[-10:])
            relative_var = recent_var / (state.loss_ema**2 + 1e-8)
            state.loss_var_ema = 0.1 * relative_var + 0.9 * state.loss_var_ema

        # Track best loss for plateau detection
        if state.loss_ema < state.best_loss_ema:
            state.best_loss_ema = state.loss_ema
            state.epochs_since_best = 0
        else:
            state.epochs_since_best += 1

        # Update cooldown counter
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            state.epochs_in_phase += 1
            return state.current_lr

        # Phase-specific logic
        state.epochs_in_phase += 1

        if state.phase == LRPhase.WARMUP:
            self._handle_warmup(epoch)
        elif state.phase == LRPhase.STABLE:
            self._handle_stable(epoch)
        elif state.phase == LRPhase.DECAY:
            self._handle_decay(epoch)

        return state.current_lr

    def _detect_instability(self) -> bool:
        """Detect if training is becoming unstable."""
        state = self.state

        # Check 1: High variance
        if state.loss_var_ema > self.variance_threshold:
            log.debug(
                f"Instability detected: variance {state.loss_var_ema:.4f} > {self.variance_threshold}"
            )
            return True

        # Check 2: Loss rebound (fast EMA significantly above slow EMA)
        if state.loss_ema_slow > 0:
            rebound_ratio = (state.loss_ema - state.loss_ema_slow) / state.loss_ema_slow
            if rebound_ratio > self.rebound_threshold:
                log.debug(
                    f"Instability detected: rebound ratio {rebound_ratio:.4f} > {self.rebound_threshold}"
                )
                return True

        return False

    def _detect_plateau(self) -> bool:
        """Detect if training has plateaued."""
        state = self.state
        return state.epochs_since_best >= self.patience

    def _detect_steady_decrease(self) -> bool:
        """Detect if loss is decreasing steadily."""
        state = self.state
        # Loss is decreasing if fast EMA is below slow EMA
        return state.loss_ema < state.loss_ema_slow * 0.99

    def _handle_warmup(self, epoch: int):
        """Handle warmup phase logic."""
        state = self.state

        # Check if we should exit warmup
        if state.epochs_in_phase >= self.min_warmup_epochs:
            if self._detect_instability():
                # Instability detected - transition to decay
                log.info(f"Epoch {epoch}: Warmup -> Decay (instability detected)")
                self._transition_to_decay(epoch)
                return

            if state.current_lr >= self.lr_max:
                # Reached max LR - transition to stable
                log.info(f"Epoch {epoch}: Warmup -> Stable (reached max LR)")
                self._transition_to_stable(epoch)
                return

        # Continue warmup - increase LR
        if state.current_lr < self.lr_max:
            state.current_lr = min(state.current_lr * self.warmup_rate, self.lr_max)

    def _handle_stable(self, epoch: int):
        """Handle stable phase logic."""
        state = self.state

        # Check for instability or plateau
        if self._detect_instability():
            log.info(f"Epoch {epoch}: Stable -> Decay (instability detected)")
            self._transition_to_decay(epoch)
        elif self._detect_plateau():
            log.info(f"Epoch {epoch}: Stable -> Decay (plateau detected)")
            self._transition_to_decay(epoch)

    def _handle_decay(self, epoch: int):
        """Handle decay phase logic."""
        state = self.state

        # Check if we've stabilized and can return to stable phase
        if (
            state.epochs_in_phase >= self.cooldown_epochs
            and self._detect_steady_decrease()
            and not self._detect_instability()
        ):
            log.info(f"Epoch {epoch}: Decay -> Stable (loss stabilized)")
            self._transition_to_stable(epoch)
            return

        # Check if we should decay further
        if state.epochs_in_phase >= self.cooldown_epochs and (
            self._detect_instability() or self._detect_plateau()
        ):
            self._reduce_lr(epoch)

    def _transition_to_stable(self, epoch: int):
        """Transition to stable phase."""
        state = self.state
        state.phase = LRPhase.STABLE
        state.phase_start_epoch = epoch
        state.epochs_in_phase = 0
        state.warmup_complete = True

    def _transition_to_decay(self, epoch: int):
        """Transition to decay phase and reduce LR."""
        state = self.state
        state.phase = LRPhase.DECAY
        state.phase_start_epoch = epoch
        state.epochs_in_phase = 0
        state.warmup_complete = True
        self._reduce_lr(epoch)

    def _reduce_lr(self, epoch: int):
        """Reduce learning rate."""
        state = self.state

        if state.total_lr_reductions >= self.max_lr_reductions:
            log.warning(f"Epoch {epoch}: Max LR reductions ({self.max_lr_reductions}) reached")
            return

        old_lr = state.current_lr
        state.current_lr = max(state.current_lr * self.decay_factor, self.lr_min)
        state.total_lr_reductions += 1

        # Reset tracking after LR change
        state.best_loss_ema = state.loss_ema
        state.epochs_since_best = 0
        self._cooldown_counter = self.cooldown_epochs

        log.info(
            f"Epoch {epoch}: LR reduced {old_lr:.2e} -> {state.current_lr:.2e} "
            f"(reduction #{state.total_lr_reductions})"
        )

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.state.current_lr

    def get_phase(self) -> str:
        """Get current phase name."""
        return self.state.phase.value

    def get_stats(self) -> dict:
        """Get scheduler statistics for logging."""
        state = self.state
        return {
            "adaptive_lr/learning_rate": state.current_lr,
            "adaptive_lr/phase": state.phase.value,
            "adaptive_lr/loss_ema": state.loss_ema,
            "adaptive_lr/loss_ema_slow": state.loss_ema_slow,
            "adaptive_lr/loss_var_ema": state.loss_var_ema,
            "adaptive_lr/epochs_since_best": state.epochs_since_best,
            "adaptive_lr/total_lr_reductions": state.total_lr_reductions,
        }


def create_adaptive_schedule(
    lr_start: float = 1e-7, lr_max: float = 1e-4, lr_min: float = 1e-8, **kwargs
) -> tuple[AdaptiveLRScheduler, optax.Schedule]:
    """
    Create an adaptive LR scheduler and a compatible optax schedule.

    The optax schedule is a placeholder that returns 1.0, and the actual
    LR is controlled by scaling the optimizer externally.

    Returns:
        Tuple of (AdaptiveLRScheduler, optax.Schedule)
    """
    scheduler = AdaptiveLRScheduler(lr_start=lr_start, lr_max=lr_max, lr_min=lr_min, **kwargs)

    # The optax schedule just returns 1.0 - we'll scale externally
    optax_schedule = optax.constant_schedule(1.0)

    return scheduler, optax_schedule


# =============================================================================
# Alternative: Reduce-on-Plateau style scheduler (simpler)
# =============================================================================


@dataclass
class ReduceOnPlateauState:
    """State for reduce-on-plateau scheduler."""

    current_lr: float
    best_loss: float = float("inf")
    epochs_since_best: int = 0
    total_reductions: int = 0
    in_cooldown: bool = False
    cooldown_counter: int = 0


class ReduceOnPlateauScheduler:
    """
    Simple reduce-on-plateau learning rate scheduler.

    Reduces LR by a factor when loss doesn't improve for `patience` epochs.
    This is simpler than the full adaptive scheduler and often works well.

    Parameters:
        lr_initial: Starting learning rate
        lr_min: Minimum learning rate
        factor: Factor to multiply LR by on reduction
        patience: Epochs to wait before reducing
        threshold: Minimum change to qualify as improvement
        cooldown: Epochs to wait after reduction before allowing another
        max_reductions: Maximum number of reductions allowed
    """

    def __init__(
        self,
        lr_initial: float = 1e-4,
        lr_min: float = 1e-8,
        factor: float = 0.5,
        patience: int = 100,
        threshold: float = 1e-4,  # Relative improvement threshold
        cooldown: int = 50,
        max_reductions: int = 10,
    ):
        self.lr_initial = lr_initial
        self.lr_min = lr_min
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.max_reductions = max_reductions

        self.state = ReduceOnPlateauState(current_lr=lr_initial)

    def update(self, loss: float, epoch: int) -> float:
        """Update scheduler with new loss and return current LR."""
        state = self.state

        # Handle cooldown
        if state.in_cooldown:
            state.cooldown_counter -= 1
            if state.cooldown_counter <= 0:
                state.in_cooldown = False
            return state.current_lr

        # Check for improvement
        if loss < state.best_loss * (1 - self.threshold):
            state.best_loss = loss
            state.epochs_since_best = 0
        else:
            state.epochs_since_best += 1

        # Check if we should reduce
        if (state.epochs_since_best >= self.patience) and (
            state.total_reductions < self.max_reductions
        ):
            old_lr = state.current_lr
            state.current_lr = max(state.current_lr * self.factor, self.lr_min)
            state.total_reductions += 1
            state.epochs_since_best = 0
            state.best_loss = loss  # Reset best loss after reduction
            state.in_cooldown = True
            state.cooldown_counter = self.cooldown

            log.info(
                f"Epoch {epoch}: ReduceOnPlateau LR {old_lr:.2e} -> {state.current_lr:.2e} "
                f"(reduction #{state.total_reductions})"
            )

        return state.current_lr

    def get_lr(self) -> float:
        return self.state.current_lr

    def get_stats(self) -> dict:
        state = self.state
        return {
            "plateau_lr/learning_rate": state.current_lr,
            "plateau_lr/best_loss": state.best_loss,
            "plateau_lr/epochs_since_best": state.epochs_since_best,
            "plateau_lr/total_reductions": state.total_reductions,
        }
