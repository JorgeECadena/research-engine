"""OpenQASM 3 export of a ``circuit/1`` document, read by this package's reader.

The exporter is deliberately downstream of :func:`quantum_sim.formats.from_gate_list`: it never
interprets a gate label, a control or an angle itself. A document the simulator refuses is refused
here too, and nothing is marked ``"ok"`` that the simulator would not run. The output uses only
gates that ``stdgates.inc`` defines; a gate with no exact spelling is refused rather than guessed
at.

The rules are the rules of the reader plus three format rules:

* a gate with an open control is exported as ``x`` before and after the operation, which is exact
  for every gate and keeps the open control from becoming a closed one;
* a free angle is rendered as an OpenQASM expression, not rounded to a decimal; a symbol with no
  binding refuses that part;
* a ``|1>`` drawn at an input (a wire legend or a text item) refuses the document: the export does
  not add an initialization that the drawing did not state as an operation.

The standard library has no ``sxdg``; it is spelled ``inv @ sx``, which is exact. Gates the
simulator knows but ``stdgates.inc`` does not define (``rxx``, ``ryy``, ``rzz``, ``rk``,
``iswap``) are refused until v1 has an exact spelling for them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from quantum_sim.angles import Angle, format_number, parse_angle
from quantum_sim.domain import MEASURE, AngleError, RefusedError
from quantum_sim.formats import ParametricCircuit, ParametricOp, Reading, from_gate_list

__all__ = ["QASM_VERSION", "QasmExport", "QasmProgram", "to_qasm"]

QASM_VERSION = 1

# A caller may write ``pi/4`` as a binding value; the angle grammar spells the constant ``\pi``.
_PI = re.compile(r"(?<![\\A-Za-z])pi(?![A-Za-z])")

# Gates whose name is defined by the OpenQASM 3 standard library. ``sxdg`` is not here: it is
# written as ``inv @ sx`` below. The check in the tests keeps this table honest.
_STDGATES = frozenset(
    {"id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "p", "swap"}
)

# Simulator gates with no exact ``stdgates.inc`` spelling in v1. Refusing is better than a
# program that means something else.
_NO_STANDARD_FORM = {
    "rxx": "stdgates.inc has no rxx, and v1 does not decompose it",
    "ryy": "stdgates.inc has no ryy, and v1 does not decompose it",
    "rzz": "stdgates.inc has no rzz, and v1 does not decompose it",
    "rk": "stdgates.inc has no R_k, and v1 does not decompose it",
    "iswap": "stdgates.inc has no iSWAP, and v1 does not decompose it",
}

# A controlled standard gate whose one-control spelling has its own name in stdgates.inc.
_ONE_CONTROL = {
    "x": "cx",
    "y": "cy",
    "z": "cz",
    "h": "ch",
    "p": "cp",
    "rx": "crx",
    "ry": "cry",
    "rz": "crz",
    "swap": "cswap",
}


@dataclass(frozen=True)
class QasmProgram:
    """One OpenQASM 3 program: a circuit part, its drawn sign, and the text."""

    part: int
    sign: str | None
    text: str
    n_qubits: int
    n_clbits: int


@dataclass(frozen=True)
class QasmExport:
    """The result of exporting one document. ``status`` is ``"ok"`` or ``"refused"``.

    A refused document has no programs: a drawing with one part the exporter cannot express is
    refused as a whole, so no side of a drawn equality is written alone. ``reasons`` names each
    refused part (the reader's reasons when the simulator refused the document).
    """

    status: str
    programs: tuple[QasmProgram, ...]
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.reasons


class _Bad(Exception):
    """One circuit part cannot be exported; the message is the reason given to the caller."""


def to_qasm(
    document: object,
    *,
    values: Mapping[str, str | float | int] | None = None,
    origin: str = "",
) -> QasmExport:
    """Read ``document`` with :func:`from_gate_list` and export it as OpenQASM 3.

    ``values`` binds a free parameter by its canonical name (``theta`` or ``\\theta``) to a LaTeX
    expression or a number. ``origin`` is a one-line provenance comment. The function returns
    refusals as data; it raises only for a programming error.
    """
    try:
        reading = from_gate_list(document)
    except RefusedError as error:
        return QasmExport("refused", (), error.reasons)
    try:
        bindings = _bindings(values)
    except _Bad as exc:
        return QasmExport("refused", (), (str(exc),))
    inputs = [bit for _, bit in reading.input_kets]
    if "1" in inputs:
        return QasmExport(
            "refused",
            (),
            ("the drawing starts an input in |1>, which the export does not add",),
        )
    return _export(reading, bindings, origin)


def _export(reading: Reading, bindings: Mapping[str, Angle], origin: str) -> QasmExport:
    programs: list[QasmProgram] = []
    reasons: list[str] = []
    total = len(reading.circuits)
    for index, circuit in enumerate(reading.circuits, 1):
        sign = reading.signs[index - 2] if index > 1 else None
        if not sign:
            sign = None
        try:
            body, n_clbits = _part(circuit, bindings)
        except _Bad as exc:
            reasons.append(f"part {index}: {exc}")
            continue
        text = _render(
            circuit,
            n_clbits,
            body=body,
            origin=f"{origin}, part {index}/{total}" if origin else "",
            sign=sign,
        )
        programs.append(QasmProgram(index, sign, text, circuit.n_wires, n_clbits))
    if reasons:
        return QasmExport("refused", (), tuple(reasons))
    return QasmExport("ok", tuple(programs), ())


# ------------------------------------------------------------------ one part


def _part(circuit: ParametricCircuit, bindings: Mapping[str, Angle]) -> tuple[list[str], int]:
    """The operation lines of one part and its classical bit count (0 without a measurement)."""
    body: list[str] = []
    measured: set[int] = set()
    for op in circuit.ops:
        wires = {*op.wires, *op.controls, *op.open_controls}
        if op.gate == MEASURE:
            if op.controls or op.open_controls:
                raise _Bad("a controlled measurement is not exported")
            if len(op.wires) != 1:
                raise _Bad("a measurement of several wires is not exported")
            wire = op.wires[0]
            if wire in measured:
                raise _Bad(f"wire {wire} is measured twice")
            measured.add(wire)
            body.append(f"measure q[{wire - 1}] -> c[{wire - 1}];")
            continue
        if measured & wires:
            raise _Bad("an operation touches a wire after its measurement")
        body.extend(_op(op, bindings))
    return body, circuit.n_wires if measured else 0


def _op(op: ParametricOp, bindings: Mapping[str, Angle]) -> list[str]:
    name = op.gate
    if name in _NO_STANDARD_FORM:
        raise _Bad(f"gate {name!r}: {_NO_STANDARD_FORM[name]}")
    if name == "sxdg":
        base = "inv @ sx"
        params: tuple[str, ...] = ()
    elif name in _STDGATES:
        params = _params(op, bindings)
        base = name + _arguments(params)
    else:  # pragma: no cover - the reader only produces gates from the simulator table
        raise _Bad(f"gate {name!r} is not in the simulator's table")
    return [_emit(op, base, params)]


def _params(op: ParametricOp, bindings: Mapping[str, Angle]) -> tuple[str, ...]:
    unbound: set[str] = set()
    rendered = tuple(_angle(angle, bindings, unbound) for angle in op.params)
    if unbound:
        names = ", ".join(sorted(unbound))
        raise _Bad(f"unbound parameter(s): {names}; bind them with a value")
    return rendered


def _arguments(params: tuple[str, ...]) -> str:
    return f"({', '.join(params)})" if params else ""


def _emit(op: ParametricOp, base: str, params: tuple[str, ...]) -> str:
    """One operation; open controls are conjugated with X, which leaves the gate exact."""
    wrapper = [f"x q[{wire - 1}];" for wire in op.open_controls]
    all_controls = (*op.controls, *op.open_controls)
    count = len(all_controls)
    if count == 0:
        head = base
    elif op.gate == "sxdg":
        head = "ctrl @ " * count + base
    elif count == 1 and op.gate in _ONE_CONTROL:
        head = _ONE_CONTROL[op.gate] + _arguments(params)
    elif op.gate == "x" and count == 2:
        head = "ccx"
    else:
        head = "ctrl @ " * count + base
    wires = (*all_controls, *op.wires)
    operation = f"{head} {', '.join(f'q[{wire - 1}]' for wire in wires)};"
    return "\n".join([*wrapper, operation, *wrapper])


# ------------------------------------------------------------------ parameters


def _bindings(values: Mapping[str, str | float | int] | None) -> dict[str, Angle]:
    """Parse every caller binding once; a value may not name another symbol."""
    bound: dict[str, Angle] = {}
    for key, value in (values or {}).items():
        name = key.strip()
        if name.startswith("{") and name.endswith("}"):
            name = name[1:-1].strip()
        name = name.lstrip("\\")
        if not name:
            raise _Bad("a parameter binding has an empty name")
        try:
            if isinstance(value, bool):
                raise AngleError("a parameter value cannot be true or false")
            angle = (
                Angle.of(float(value))
                if isinstance(value, int | float)
                else parse_angle(_PI.sub(r"\\pi", str(value)))
            )
        except AngleError as error:
            raise _Bad(f"value for {name!r}: {error}") from None
        if angle.symbols:
            names = ", ".join(angle.symbols)
            raise _Bad(f"value for {name!r} has free symbol(s): {names}")
        bound[name] = angle
    return bound


def _angle(angle: Angle, bindings: Mapping[str, Angle], unbound: set[str]) -> str:
    return _strip(_node(angle.tree, bindings, unbound))


def _node(node: Any, bindings: Mapping[str, Angle], unbound: set[str]) -> str:
    kind = node[0]
    text: str
    if kind == "num":
        value = format_number(float(node[1]))
        if value.endswith(".0") and not value.startswith("-0.0"):
            value = value[:-2]
        text = f"({value})" if value.startswith("-") else value
    elif kind == "pi":
        text = "pi"
    elif kind == "sym":
        name = node[1]
        bound = bindings.get(name)
        if bound is None:
            unbound.add(name)
            text = name
        else:
            text = _node(bound.tree, bindings, unbound)
    elif kind == "neg":
        text = f"(-{_node(node[1], bindings, unbound)})"
    elif kind == "sqrt":
        text = f"sqrt({_node(node[1], bindings, unbound)})"
    elif kind == "pow":
        text = f"({_node(node[1], bindings, unbound)}**{node[2]})"
    else:
        left, right = _node(node[1], bindings, unbound), _node(node[2], bindings, unbound)
        operator = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[kind]
        text = f"({left}{operator}{right})"
    return text


def _strip(value: str) -> str:
    """Drop one layer of parentheses only when it wraps the whole expression."""
    if len(value) < 2 or value[0] != "(" or value[-1] != ")":
        return value
    depth = 0
    for position, character in enumerate(value):
        depth += character == "("
        depth -= character == ")"
        if depth == 0 and position < len(value) - 1:
            return value
    return value[1:-1]


# ------------------------------------------------------------------ text


def _render(
    circuit: ParametricCircuit,
    n_clbits: int,
    *,
    body: list[str],
    origin: str,
    sign: str | None,
) -> str:
    n_qubits = circuit.n_wires
    lines = [f"// Generated by docingest (qasm format {QASM_VERSION})"]
    if origin:
        lines.append(f"// source: {_comment(origin)}")
    if sign is not None:
        lines.append(f"// sign: {_comment(sign)}")
    lines.append("// wire 1 (top of the drawing) = q[0]; OpenQASM fixes no bit order")
    legends = "; ".join(_comment(wire) for wire in circuit.wire_labels if wire.strip())
    if legends:
        lines.append(f"// wires: {legends}")
    lines.extend(["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n_qubits}] q;"])
    if n_clbits:
        lines.append(f"bit[{n_clbits}] c;")
    lines.append("")
    return "\n".join([*lines, *body]) + "\n"


def _comment(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").strip()
