"""Quantum-sim adapter for the circuit exporter port.

This is the only docingest module that imports ``quantum_sim`` (an import-linter contract checks
it). ``bootstrap`` imports it lazily, so a command that never exports a circuit does not load
``quantum_sim``. Binding names follow ``quantum_sim.angles.symbol_name``, the one rule: the
adapter uses it to report, for a refused circuit, the symbols the caller gave no value.
"""

from __future__ import annotations

from collections.abc import Mapping

from quantum_sim.angles import symbol_name
from quantum_sim.qasm import to_qasm, validate_bindings

from ..domain.circuits import CircuitSteps
from ..ports import QasmExport, QasmProgram

__all__ = ["QuantumSimExporter"]


class QuantumSimExporter:
    """Emit OpenQASM 3 with ``quantum_sim.qasm``, after the simulator reads the circuit."""

    def check_values(self, values: Mapping[str, str | float | int] | None = None) -> None:
        validate_bindings(values)

    def export(
        self,
        circuit: CircuitSteps,
        *,
        values: Mapping[str, str | float | int] | None = None,
        origin: str = "",
        tool: str = "docingest",
    ) -> QasmExport:
        export = to_qasm(circuit.to_dict(), values=values, origin=origin, tool=tool)
        unbound: tuple[str, ...] = ()
        if not export.ok:
            named = {symbol_name(key) for key in values or {}}
            unbound = tuple(symbol for symbol in export.symbols if symbol not in named)
        return QasmExport(
            status=export.status,
            programs=tuple(
                QasmProgram(
                    part=program.part,
                    sign=program.sign,
                    text=program.text,
                    n_qubits=program.n_qubits,
                    n_clbits=program.n_clbits,
                )
                for program in export.programs
            ),
            reasons=export.reasons,
            notes=export.notes,
            symbols=export.symbols,
            unmatched=export.unmatched,
            unbound=unbound,
        )
