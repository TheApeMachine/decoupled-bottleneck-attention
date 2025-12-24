"""
trainer provides the training loop.
"""

from __future__ import annotations

from caramba.config.manifest import Manifest
from caramba.config.mode import Mode
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
            print(f"trainer: group={group.name!r} runs={len(group.runs)}")
            session: Upcycle | None = None
            for run in group.runs:
                print(
                    "trainer: run "
                    f"id={run.id!r} mode={run.mode} steps={run.steps}"
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
