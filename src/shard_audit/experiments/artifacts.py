"""Canonical locations and metadata for generated experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    """One self-contained experiment run under the repository artifact root."""

    root: Path
    experiment_id: str
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.root / self.experiment_id / self.run_id

    @property
    def prepared(self) -> Path:
        return self.run_root / "prepared"

    @property
    def scores(self) -> Path:
        return self.run_root / "scores"

    @property
    def results(self) -> Path:
        return self.run_root / "results"

    @property
    def reports(self) -> Path:
        return self.run_root / "reports"

    @property
    def logs(self) -> Path:
        return self.run_root / "logs"

    def create(self) -> None:
        for path in (self.prepared, self.scores, self.results, self.reports, self.logs):
            path.mkdir(parents=True, exist_ok=True)


def artifact_layout(repo_root: str | Path, experiment_id: str, run_id: str) -> ArtifactLayout:
    """Return the canonical location for a named experiment run."""
    return ArtifactLayout(Path(repo_root) / "artifacts", experiment_id, run_id)
