"""
trainer provides the training loop.
"""

from __future__ import annotations

from caramba.config.manifest import Manifest
from caramba.config.mode import Mode
from caramba.console import logger
from caramba.trainer.upcycle import Upcycle


class Trainer:
    """
    Trainer composes the training loop.
    """

    def __init__(
        self,
        manifest: Manifest,
    ) -> None:
        self.manifest: Manifest = manifest

    def run(self) -> None:
        """
        Run the training loop.
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
