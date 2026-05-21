"""ANDES adapter for benchmark loading and model/DAE introspection.

This module is the only package layer that directly imports ANDES. It keeps the
rest of the implementation independent of ANDES internals while exposing the
quantities needed by the paper: benchmark cases, model families, device
ownership and status, variable addresses, and DAE Jacobian blocks.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

from .common import project_root_from


class AndesAdapterError(RuntimeError):
    """Raised when ANDES loading or introspection fails."""


@dataclass(frozen=True, slots=True)
class ResolvedCase:
    """Concrete case path and optional companion dynamic file."""

    case_id: str
    case_path: Path
    addfile_path: Path | None = None
    source: str = "unknown"
    description: str | None = None

    @property
    def exists(self) -> bool:
        return self.case_path.exists() and (
            self.addfile_path is None or self.addfile_path.exists()
        )


@dataclass(frozen=True, slots=True)
class AndesCaseSpec:
    """User-facing benchmark loading request."""

    case: str | Path
    addfile: str | Path | None = None
    setup: bool = True
    run_pflow: bool = False
    init_tds: bool = False
    default_config: bool = True
    pycode_path: str | Path | None = None
    input_path: str | Path | None = None
    load_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VariableAddress:
    """Address of one ANDES variable for one device."""

    model: str
    variable: str
    device_id: str
    device_position: int
    address: int
    domain: str
    variable_class: str


@dataclass(frozen=True, slots=True)
class DeviceRecord:
    """One loaded ANDES device."""

    model: str
    device_id: str
    position: int
    status: float | None
    bus: str | None = None
    owners: Mapping[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """Summary of one ANDES model class in the loaded system."""

    name: str
    group: str
    n: int
    class_name: str
    family: str
    explicit_control_supported: bool
    devices: tuple[DeviceRecord, ...]


@dataclass(frozen=True, slots=True)
class ModelFamilyRegistry:
    """Registry of ANDES model families present in a loaded system."""

    families: Mapping[str, tuple[str, ...]]
    model_to_family: Mapping[str, str]
    explicit_control_models: tuple[str, ...]
    generic_models: tuple[str, ...]
    unknown_models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JacobianBlocks:
    """ANDES DAE Jacobian blocks."""

    fx: sparse.csc_matrix
    fy: sparse.csc_matrix
    gx: sparse.csc_matrix
    gy: sparse.csc_matrix
    tf: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.fx.shape[0])

    @property
    def n_algebraic(self) -> int:
        return int(self.gy.shape[0])


STOCK_CASE_ALIASES: dict[str, tuple[str, str | None, str]] = {
    "ieee14": ("ieee14/ieee14.raw", None, "Stock IEEE-14 power-flow case"),
    "ieee14_full": ("ieee14/ieee14_full.xlsx", None, "Stock IEEE-14 dynamic case"),
    "ieee39": ("ieee39/ieee39_full.xlsx", None, "Stock IEEE-39 dynamic case"),
    "ieee39_full": ("ieee39/ieee39_full.xlsx", None, "Stock IEEE-39 dynamic case"),
    "ieee39_raw": ("ieee39/ieee39.raw", None, "Stock IEEE-39 RAW case"),
    "npcc": ("npcc/npcc.xlsx", None, "Stock NPCC static case"),
    "npcc_full": ("npcc/npcc.raw", "npcc/npcc_full.dyr", "Stock NPCC RAW/DYR case"),
    "gbnetwork": ("GBnetwork/GBnetwork.xlsx", None, "Stock GBnetwork XLSX case"),
    "gbnetwork_m": ("GBnetwork/GBnetwork.m", None, "Stock GBnetwork MATPOWER case"),
    "wecc": ("wecc/wecc.raw", "wecc/wecc_full.dyr", "Stock WECC RAW/DYR case"),
}


LOCAL_CASE_ALIASES: dict[str, tuple[str, str | None, str]] = {
    "activsg2000_stable": (
        "data/activsg2000_stable/ACTIVSg2000.RAW",
        "data/activsg2000_stable/ACTIVSg2000_dynamics.dyr",
        "Local ACTIVSg2000 trimmed stable RAW/DYR case",
    ),
}


EXPLICIT_GROUP_FAMILIES: dict[str, str] = {
    "ACNode": "buses",
    "ACLine": "lines",
    "ACTopology": "topology",
    "StaticGen": "static_generators",
    "StaticLoad": "loads",
    "StaticShunt": "shunts",
    "SynGen": "synchronous_machines",
    "Exciter": "exciters",
    "TurbineGov": "governors",
    "PSS": "pss",
    "DynLoad": "dynamic_loads",
    "Motor": "motors",
    "RenGen": "renewable_generators",
    "RenExciter": "renewable_exciters",
    "RenGovernor": "renewable_governors",
    "RenPlant": "renewable_plant_controls",
    "RenPitch": "renewable_pitch_controls",
    "RenTorque": "renewable_torque_controls",
    "RenAerodynamics": "renewable_aerodynamics",
    "DG": "distributed_resources",
    "DGProtection": "distributed_resource_protection",
    "StaticACDC": "static_acdc",
    "DCLink": "dc_link",
    "DCTopology": "dc_topology",
    "TimedEvent": "timed_events",
    "VoltComp": "voltage_compensation",
    "PLL": "pll",
    "Calculation": "calculation",
    "FreqMeasurement": "frequency_measurement",
    "PhasorMeasurement": "phasor_measurement",
}


CONTROL_FAMILIES = {
    "exciters",
    "governors",
    "pss",
    "renewable_exciters",
    "renewable_governors",
    "renewable_plant_controls",
    "renewable_pitch_controls",
    "static_acdc",
    "dc_link",
    "voltage_compensation",
    "pll",
}


@contextmanager
def _maybe_quiet(enabled: bool):
    if not enabled:
        yield
        return
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        yield


def _import_andes():
    try:
        import andes  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise AndesAdapterError(f"ANDES import failed: {exc}") from exc
    return andes


def _project_cache(project_root: Path) -> Path:
    path = project_root / "results" / ".andes" / "pycode"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _repair_esdc1a_limit_references(system: Any) -> list[str]:
    """Repair a known ANDES ESDC1A limiter-reference issue.

    ANDES derives ESDC1A from ESDC2A and replaces ``VRU``/``VRL`` with
    constant services, but the anti-windup limiter can retain references to
    the removed ESDC2A variable services.  Those stale services are not
    evaluated and remain scalar zero, which breaks TDS initialization for
    ACTIVSg2000-style data.  Rewiring the limiter to the active ESDC1A limit
    services restores the intended ``VRMAX``/``VRMIN`` limits.
    """

    if not hasattr(system, "ESDC1A"):
        return []
    model = system.ESDC1A
    if int(getattr(model, "n", 0)) <= 0:
        return []
    upper = getattr(model, "VRU", None)
    lower = getattr(model, "VRL", None)
    if upper is None or lower is None:
        return []

    repaired: list[str] = []
    for attr in ("LA", "LA_lim"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        changed = False
        if hasattr(obj, "upper") and getattr(obj, "upper", None) is not upper:
            obj.upper = upper
            changed = True
        if hasattr(obj, "lower") and getattr(obj, "lower", None) is not lower:
            obj.lower = lower
            changed = True
        if changed:
            repaired.append(f"ESDC1A.{attr}")
    return repaired


def _candidate_local_roots(project_root: Path) -> tuple[Path, ...]:
    roots = [
        project_root,
        project_root.parent / "overvoltage_cascades",
        Path.home() / "Desktop" / "overvoltage_cascades",
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return tuple(deduped)


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def resolve_case(case: str | Path, project_root: str | Path = ".") -> ResolvedCase:
    """Resolve a stock alias, local alias, or explicit path to a case."""

    root = project_root_from(project_root)
    andes = _import_andes()
    case_str = str(case)
    key = case_str.lower()

    if key in STOCK_CASE_ALIASES:
        rel_case, rel_addfile, description = STOCK_CASE_ALIASES[key]
        case_path = Path(andes.get_case(rel_case)).resolve()
        addfile = Path(andes.get_case(rel_addfile)).resolve() if rel_addfile else None
        return ResolvedCase(key, case_path, addfile, source="andes_stock", description=description)

    if key in LOCAL_CASE_ALIASES:
        rel_case, rel_addfile, description = LOCAL_CASE_ALIASES[key]
        for local_root in _candidate_local_roots(root):
            candidate = local_root / rel_case
            addfile = local_root / rel_addfile if rel_addfile else None
            if candidate.exists() and (addfile is None or addfile.exists()):
                return ResolvedCase(
                    key,
                    candidate.resolve(),
                    addfile.resolve() if addfile else None,
                    source="local",
                    description=description,
                )
        first_root = _candidate_local_roots(root)[0]
        return ResolvedCase(
            key,
            (first_root / rel_case).resolve(),
            (first_root / rel_addfile).resolve() if rel_addfile else None,
            source="local_missing",
            description=description,
        )

    explicit = _resolve_path(case, root)
    return ResolvedCase(explicit.stem, explicit, source="explicit_path")


def available_benchmarks(project_root: str | Path = ".") -> dict[str, ResolvedCase]:
    """Return known stock and local benchmark aliases with existence flags."""

    aliases = list(STOCK_CASE_ALIASES) + list(LOCAL_CASE_ALIASES)
    return {alias: resolve_case(alias, project_root) for alias in aliases}


def _value_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _model_idx_values(model: Any) -> list[Any]:
    idx = getattr(model, "idx", None)
    return _value_list(getattr(idx, "v", []))


def _status_for(model: Any, position: int) -> float | None:
    u = getattr(model, "u", None)
    values = _value_list(getattr(u, "v", []))
    if position < len(values):
        try:
            return float(values[position])
        except Exception:
            return None
    return None


def _position_of(model: Any, device_id: str | int | float) -> int:
    requested = str(device_id)
    for position, idx in enumerate(_model_idx_values(model)):
        if idx == device_id or str(idx) == requested:
            return position
    raise AndesAdapterError(f"Device {device_id!r} not found in model {model.class_name}")


def _address_values(variable: Any) -> list[int]:
    values = _value_list(getattr(variable, "a", []))
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def _variable_domain(variable: Any) -> str:
    class_name = variable.__class__.__name__
    if "State" in class_name:
        return "state"
    if "Algeb" in class_name:
        return "algebraic"
    return "other"


def _owner_values(model: Any, position: int) -> dict[str, str | None]:
    owners: dict[str, str | None] = {}
    for attr in ("owner", "bus", "bus1", "bus2", "gen", "syn", "exc", "gov", "pss"):
        if not hasattr(model, attr):
            continue
        values = _value_list(getattr(getattr(model, attr), "v", []))
        if position < len(values):
            value = values[position]
            owners[attr] = None if value is None else str(value)
    return owners


def _family_for_group(group: str | None) -> str:
    if not group:
        return "unknown"
    return EXPLICIT_GROUP_FAMILIES.get(group, f"generic:{group}")


def _kvxopt_to_csc(matrix: Any) -> sparse.csc_matrix:
    """Convert kvxopt/sparse-like matrices to SciPy CSC."""

    if sparse.issparse(matrix):
        return matrix.tocsc()
    size = getattr(matrix, "size", None)
    if size is not None and hasattr(matrix, "I") and hasattr(matrix, "J") and hasattr(matrix, "V"):
        rows, cols = int(size[0]), int(size[1])
        data = np.asarray(matrix.V, dtype=float).reshape(-1)
        row = np.asarray(matrix.I, dtype=int).reshape(-1)
        col = np.asarray(matrix.J, dtype=int).reshape(-1)
        return sparse.coo_matrix((data, (row, col)), shape=(rows, cols)).tocsc()
    array = np.asarray(matrix, dtype=float)
    return sparse.csc_matrix(array)


class AndesCase:
    """Wrapper around an ANDES ``System`` with stable accessors."""

    def __init__(
        self,
        spec: AndesCaseSpec,
        *,
        project_root: str | Path = ".",
        quiet: bool = True,
    ) -> None:
        self.project_root = project_root_from(project_root)
        self.spec = spec
        self.resolved = resolve_case(spec.case, self.project_root)
        if spec.addfile is not None:
            self.resolved = ResolvedCase(
                self.resolved.case_id,
                self.resolved.case_path,
                _resolve_path(spec.addfile, self.project_root),
                self.resolved.source,
                self.resolved.description,
            )
        self.quiet = quiet
        self.system: Any | None = None
        self.is_loaded = False
        self.is_setup = False
        self.pflow_ran = False
        self.tds_initialized = False

    @classmethod
    def load(
        cls,
        spec: AndesCaseSpec | str | Path,
        *,
        project_root: str | Path = ".",
        quiet: bool = True,
    ) -> "AndesCase":
        if not isinstance(spec, AndesCaseSpec):
            spec = AndesCaseSpec(spec)
        case = cls(spec, project_root=project_root, quiet=quiet)
        case.load_system()
        return case

    def load_system(self) -> Any:
        """Load the ANDES system according to the case spec."""

        if not self.resolved.exists:
            raise AndesAdapterError(f"Case path does not exist: {self.resolved}")

        andes = _import_andes()
        pycode_path = self.spec.pycode_path
        if pycode_path is None:
            pycode_path = _project_cache(self.project_root)
        else:
            pycode_path = _resolve_path(pycode_path, self.project_root)
            pycode_path.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "setup": self.spec.setup,
            "default_config": self.spec.default_config,
            "pycode_path": str(pycode_path),
            "no_output": True,
        }
        if self.resolved.addfile_path is not None:
            kwargs["addfile"] = str(self.resolved.addfile_path)
        if self.spec.input_path is not None:
            kwargs["input_path"] = str(_resolve_path(self.spec.input_path, self.project_root))
        kwargs.update(dict(self.spec.load_kwargs))

        with _maybe_quiet(self.quiet):
            system = andes.load(str(self.resolved.case_path), **kwargs)
        if system is None:
            raise AndesAdapterError(f"ANDES returned None while loading {self.resolved.case_path}")
        _repair_esdc1a_limit_references(system)

        self.system = system
        self.is_loaded = True
        self.is_setup = bool(self.spec.setup)
        if self.spec.run_pflow:
            self.run_pflow()
        if self.spec.init_tds:
            self.init_tds()
        return system

    def require_system(self) -> Any:
        if self.system is None:
            raise AndesAdapterError("ANDES case has not been loaded")
        return self.system

    def setup(self) -> None:
        system = self.require_system()
        if not self.is_setup:
            with _maybe_quiet(self.quiet):
                system.setup()
            self.is_setup = True

    def run_pflow(self) -> bool:
        self.setup()
        system = self.require_system()
        with _maybe_quiet(self.quiet):
            result = bool(system.PFlow.run())
        self.pflow_ran = result
        return result

    def init_tds(self) -> None:
        self.setup()
        if not self.pflow_ran:
            self.run_pflow()
        system = self.require_system()
        with _maybe_quiet(self.quiet):
            system.TDS.init()
        self.tds_initialized = True

    @property
    def base_mva(self) -> float | None:
        system = self.require_system()
        config = getattr(system, "config", None)
        if config is not None and hasattr(config, "mva"):
            try:
                return float(config.mva)
            except Exception:
                pass
        try:
            return float(system.config.mva)
        except Exception:
            return None

    def get_model(self, model_name: str) -> Any:
        system = self.require_system()
        if model_name not in system.models:
            raise AndesAdapterError(f"Model {model_name!r} is not available in this system")
        return system.models[model_name]

    def model_names(self, *, nonempty: bool = False) -> tuple[str, ...]:
        system = self.require_system()
        names = []
        for name, model in system.models.items():
            if nonempty and int(getattr(model, "n", 0)) == 0:
                continue
            names.append(name)
        return tuple(names)

    def device_records(self, model_name: str) -> tuple[DeviceRecord, ...]:
        model = self.get_model(model_name)
        records: list[DeviceRecord] = []
        for position, idx in enumerate(_model_idx_values(model)):
            owners = _owner_values(model, position)
            records.append(
                DeviceRecord(
                    model=model_name,
                    device_id=str(idx),
                    position=position,
                    status=_status_for(model, position),
                    bus=owners.get("bus"),
                    owners=owners,
                )
            )
        return tuple(records)

    def model_record(self, model_name: str) -> ModelRecord:
        model = self.get_model(model_name)
        group = str(getattr(model, "group", "") or "")
        family = _family_for_group(group)
        return ModelRecord(
            name=model_name,
            group=group,
            n=int(getattr(model, "n", 0)),
            class_name=str(getattr(model, "class_name", model.__class__.__name__)),
            family=family,
            explicit_control_supported=family in CONTROL_FAMILIES,
            devices=self.device_records(model_name),
        )

    def model_records(self, *, nonempty: bool = False) -> tuple[ModelRecord, ...]:
        return tuple(self.model_record(name) for name in self.model_names(nonempty=nonempty))

    def registry(self) -> ModelFamilyRegistry:
        family_map: dict[str, list[str]] = {}
        model_to_family: dict[str, str] = {}
        explicit: list[str] = []
        generic: list[str] = []
        unknown: list[str] = []

        for record in self.model_records(nonempty=False):
            family_map.setdefault(record.family, []).append(record.name)
            model_to_family[record.name] = record.family
            if record.explicit_control_supported:
                explicit.append(record.name)
            elif record.family == "unknown":
                unknown.append(record.name)
            else:
                generic.append(record.name)

        return ModelFamilyRegistry(
            families={key: tuple(value) for key, value in sorted(family_map.items())},
            model_to_family=model_to_family,
            explicit_control_models=tuple(explicit),
            generic_models=tuple(generic),
            unknown_models=tuple(unknown),
        )

    def variable_address(
        self,
        model_name: str,
        variable_name: str,
        device_id: str | int | float,
    ) -> VariableAddress:
        model = self.get_model(model_name)
        if not hasattr(model, variable_name):
            raise AndesAdapterError(f"{model_name}.{variable_name} does not exist")
        variable = getattr(model, variable_name)
        addresses = _address_values(variable)
        position = _position_of(model, device_id)
        if position >= len(addresses):
            raise AndesAdapterError(
                f"{model_name}.{variable_name} has no address for device {device_id!r}"
            )
        device = _model_idx_values(model)[position]
        return VariableAddress(
            model=model_name,
            variable=variable_name,
            device_id=str(device),
            device_position=position,
            address=addresses[position],
            domain=_variable_domain(variable),
            variable_class=variable.__class__.__name__,
        )

    def variable_addresses(self, model_name: str, variable_name: str) -> tuple[VariableAddress, ...]:
        model = self.get_model(model_name)
        if not hasattr(model, variable_name):
            raise AndesAdapterError(f"{model_name}.{variable_name} does not exist")
        variable = getattr(model, variable_name)
        addresses = _address_values(variable)
        devices = _model_idx_values(model)
        domain = _variable_domain(variable)
        return tuple(
            VariableAddress(
                model=model_name,
                variable=variable_name,
                device_id=str(device),
                device_position=position,
                address=addresses[position],
                domain=domain,
                variable_class=variable.__class__.__name__,
            )
            for position, device in enumerate(devices[: len(addresses)])
        )

    def bus_angle_address(self, bus_id: str | int | float) -> int:
        return self.variable_address("Bus", "a", bus_id).address

    def bus_voltage_address(self, bus_id: str | int | float) -> int:
        return self.variable_address("Bus", "v", bus_id).address

    def dynamic_state_address(
        self,
        model_name: str,
        variable_name: str,
        device_id: str | int | float,
    ) -> int:
        """Return a dynamic-state address and verify the variable is a state."""

        address = self.variable_address(model_name, variable_name, device_id)
        if address.domain != "state":
            raise AndesAdapterError(
                f"{model_name}.{variable_name} for {device_id!r} is not a dynamic state"
            )
        return address.address

    def algebraic_equation_address(
        self,
        model_name: str,
        variable_name: str,
        device_id: str | int | float,
    ) -> int:
        """Return an algebraic variable/equation address in the DAE y/g ordering."""

        address = self.variable_address(model_name, variable_name, device_id)
        if address.domain != "algebraic":
            raise AndesAdapterError(
                f"{model_name}.{variable_name} for {device_id!r} is not algebraic"
            )
        return address.address

    def bus_voltage_addresses(self) -> dict[str, int]:
        return {item.device_id: item.address for item in self.variable_addresses("Bus", "v")}

    def bus_angle_addresses(self) -> dict[str, int]:
        return {item.device_id: item.address for item in self.variable_addresses("Bus", "a")}

    def state_addresses(self, *, nonempty: bool = True) -> tuple[VariableAddress, ...]:
        records: list[VariableAddress] = []
        for model_name in self.model_names(nonempty=nonempty):
            model = self.get_model(model_name)
            for attr_name in dir(model):
                if attr_name.startswith("_"):
                    continue
                try:
                    variable = getattr(model, attr_name)
                except Exception:
                    continue
                if _variable_domain(variable) != "state":
                    continue
                records.extend(self.variable_addresses(model_name, attr_name))
        return tuple(records)

    def algebraic_addresses(self, *, nonempty: bool = True) -> tuple[VariableAddress, ...]:
        records: list[VariableAddress] = []
        for model_name in self.model_names(nonempty=nonempty):
            model = self.get_model(model_name)
            for attr_name in dir(model):
                if attr_name.startswith("_"):
                    continue
                try:
                    variable = getattr(model, attr_name)
                except Exception:
                    continue
                if _variable_domain(variable) != "algebraic":
                    continue
                records.extend(self.variable_addresses(model_name, attr_name))
        return tuple(records)

    def jacobians(self) -> JacobianBlocks:
        self.setup()
        system = self.require_system()
        dae = system.dae
        return JacobianBlocks(
            fx=_kvxopt_to_csc(dae.fx),
            fy=_kvxopt_to_csc(dae.fy),
            gx=_kvxopt_to_csc(dae.gx),
            gy=_kvxopt_to_csc(dae.gy),
            tf=np.asarray(getattr(dae, "Tf", np.array([])), dtype=float).reshape(-1),
        )

    def summary(self) -> dict[str, Any]:
        system = self.require_system()
        registry = self.registry()
        return {
            "case_id": self.resolved.case_id,
            "case_path": str(self.resolved.case_path),
            "addfile_path": str(self.resolved.addfile_path) if self.resolved.addfile_path else None,
            "source": self.resolved.source,
            "is_setup": self.is_setup,
            "pflow_ran": self.pflow_ran,
            "tds_initialized": self.tds_initialized,
            "n_models": len(system.models),
            "n_buses": int(getattr(system.Bus, "n", 0)) if hasattr(system, "Bus") else 0,
            "n_lines": int(getattr(system.Line, "n", 0)) if hasattr(system, "Line") else 0,
            "dae_n": int(getattr(system.dae, "n", 0)),
            "dae_m": int(getattr(system.dae, "m", 0)),
            "families": {key: len(value) for key, value in registry.families.items()},
        }
