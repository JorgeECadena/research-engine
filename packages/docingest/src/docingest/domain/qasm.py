"""Compatibility wrapper: export a paper's circuit with quantum-sim's reader and QASM writer.

docingest reads a drawing into :class:`~docingest.domain.circuits.CircuitSteps`; that class's
``to_dict()`` is exactly the ``circuit/1`` format that ``quantum_sim.formats.from_gate_list``
reads. The exporter lives in :mod:`quantum_sim.qasm`, so every document is validated by the
simulator's reader before a line is written: a circuit the simulator refuses is refused here,
and no gate, control or angle is interpreted by docingest itself.

This module stays as the stable entry point for callers of the docingest API (the web adapter
imports it); new code may call ``quantum_sim.to_qasm`` with a ``circuit/1`` document directly.
"""

from __future__ import annotations

from collections.abc import Mapping

from quantum_sim.qasm import QASM_VERSION, QasmExport, QasmProgram
from quantum_sim.qasm import to_qasm as _to_qasm

from .circuits import CircuitSteps

__all__ = ["QASM_VERSION", "QasmExport", "QasmProgram", "to_qasm"]


def to_qasm(
    circuit: CircuitSteps,
    *,
    values: Mapping[str, str | float | int] | None = None,
    origin: str = "",
) -> QasmExport:
    """Export ``circuit`` as OpenQASM 3, one program per part, through quantum-sim's reader.

    ``values`` binds a free parameter by its canonical name (``theta`` or ``\\theta``) to a LaTeX
    expression or a number. ``origin`` is a one-line provenance comment. Refusals are returned as
    data; an unparsed or ambiguous circuit is never marked ``"ok"``.
    """
    return _to_qasm(circuit.to_dict(), values=values, origin=origin)
