"""
Optimizer is an object that sits at the heart of the self-optimizing
architecture.

An optimizer is a high-level object that orchestrates optimalizations
for a process, such as training or inferencing a model.
"""
class BaseOptimizer:
    """
    BaseOptimizer provides a shared interface for all optimizers.
    """
    def __init__(self) -> None:
        self.optimizer: object | None = None

    def optimize(self, process: object) -> None:
        """
        optimize is used to orchestrate optimalizations for a process.
        """
        _ = process
        raise NotImplementedError
