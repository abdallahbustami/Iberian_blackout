"""Replication studies for the Iberian blackout paper.

The package separates the large ACTIVSg2000 physical mechanism engine from
paper-facing scenario launchers and calibration utilities. Root-level
``replication*.py`` files are compatibility wrappers only.
"""

from .activsg2000_engine import (
    ReplicationConfig,
    build_report_collector_specs,
    make_replication_plot,
    run_replication,
)

__all__ = [
    "REPORT_ANCHORS",
    "ReplicationConfig",
    "build_report_collector_specs",
    "build_academic_replica_config",
    "make_replication_plot",
    "run_replication",
]


def __getattr__(name: str):
    if name in {"REPORT_ANCHORS", "build_academic_replica_config"}:
        from . import academic_replica

        return getattr(academic_replica, name)
    raise AttributeError(name)
