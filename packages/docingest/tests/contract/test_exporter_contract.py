"""CircuitExporter contract: what ``QasmExportService`` relies on from every exporter.

The service narrows the "matches no symbol" warning with ``unmatched`` and names the circuits
refused for lack of a value with ``unbound``, so an exporter must report both as the port says.
"""

from __future__ import annotations

import pytest

from docingest.adapters.qasm import QuantumSimExporter
from docingest.domain.circuits import CircuitSteps, gate_list
from docingest.ports import CircuitExporter, QasmExport

H = "\\begin{quantikz}\n& \\gate{H} & \\qw\n\\end{quantikz}"
THETA = "\\begin{quantikz}\n& \\gate{R_y(\\theta)} & \\qw\n\\end{quantikz}"
TWO = "\\begin{quantikz}\n& \\gate{R_y(\\theta)} & \\gate{R_z(\\phi)} & \\qw\n\\end{quantikz}"


def circuit(source: str) -> CircuitSteps:
    [steps] = gate_list(source).circuits
    return steps


@pytest.fixture(params=[QuantumSimExporter], ids=["quantum-sim"])
def exporter(request) -> CircuitExporter:
    return request.param()


def test_an_exporter_is_a_circuit_exporter(exporter):
    assert isinstance(exporter, CircuitExporter)


def test_an_ok_export_has_programs_and_no_reasons(exporter):
    export = exporter.export(circuit(H), origin="a.md, block 1, circuit 1")
    assert isinstance(export, QasmExport)
    assert export.status == "ok" and export.ok and export.reasons == ()
    [program] = export.programs
    assert program.part == 1 and "h q[0];" in program.text
    assert export.unbound == ()


def test_a_key_that_matches_no_symbol_comes_back_in_unmatched(exporter):
    export = exporter.export(circuit(THETA), values={"theta": "0.5", "\\phi": "1"})
    assert export.status == "ok"
    assert export.unmatched == ("\\phi",)
    assert exporter.export(circuit(THETA), values={"\\theta": "0.5"}).unmatched == ()


def test_the_symbols_without_a_value_of_a_refused_export_are_unbound(exporter):
    export = exporter.export(circuit(TWO), values={"\\theta": "0.5"})
    assert export.status == "refused" and export.programs == ()
    assert export.symbols == ("phi", "theta")
    assert export.unbound == ("phi",)


def test_a_key_that_is_not_one_symbol_is_a_value_error(exporter):
    exporter.check_values({"theta": "pi/4", "\\theta_{1}": "1"})
    with pytest.raises(ValueError, match="is not a parameter name"):
        exporter.check_values({"thetta": "1"})
