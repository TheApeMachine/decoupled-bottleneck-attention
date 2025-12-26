"""Main training loop orchestrator.

The Trainer class reads a manifest file and executes training runs in order.
Each run can be standard training, upcycling, or evaluation. The trainer
handles session management so runs within a group share state.
"""
from __future__ import annotations

from caramba.config.manifest import Manifest
from caramba.config.mode import Mode
from caramba.console import logger
from caramba.trainer.upcycle import Upcycle


class Trainer:
    """Orchestrates training runs from a manifest.

    Iterates through groups and runs, dispatching to the appropriate
    training implementation (currently Upcycle for TRAIN mode).
    """

    def __init__(
        self,
        manifest: Manifest,
    ) -> None:
        """Set up the trainer with a manifest."""
        self.manifest: Manifest = manifest

    def run(self) -> None:
        """Execute all training runs in the manifest.

        Runs are processed in order within each group. A session (Upcycle)
        is created for the first run and reused for subsequent runs in
        the same group.
        """
        for group in self.manifest.groups:
            logger.header("Training", f"group={group.name!r}")
            logger.info(f"Scheduled {len(group.runs)} runs")
            session: Upcycle | None = None
            for run in group.runs:
                logger.step(
                    run.id if isinstance(run.id, int) else 0,
                    len(group.runs),
                    f"mode={run.mode} steps={run.steps}",
                )
                if run.mode != Mode.TRAIN:
                    raise ValueError(
                        f"Unsupported mode for run {run.id}: {run.mode}"
                    )
                if run.train is None:
                    raise ValueError(f"Run {run.id} has no train config.")

                if session is None:
                    session = Upcycle(
                        manifest=self.manifest,
                        group=group,
                        train=run.train,
                    )
                session.run(run)
