"""Export the circuits of stored documents as OpenQASM 3 programs.

The canonical store holds Markdown, and a circuit reaches it as a fenced ``latex`` block. This
use case walks those blocks, calls the :class:`~docingest.ports.CircuitExporter` port once per
circuit, and writes or removes deterministic files. The emitter and its simulator reader sit
behind that port; this service neither imports them nor interprets a gate, a control or an angle.

It never changes the store, the manifest or the cache key: every file it writes is derived and can
be made again from ``document.md``. The output directory is a sibling of the document folders
(``<output_dir>/qasm`` by default), so ``DocumentStore.corpus`` never mistakes it for a document:
it has no ``manifest.json``.

Files are written atomically with the permissions of a normal write (the umask, as the filesystem
store does). After a document's circuits are exported, its ``<doc_id[:16]>-b*-c*-p*.qasm`` files
that this run did not write are removed (a refused circuit, a lost part, block or circuit), and
the count is reported. Other documents' programs, other files and a document whose Markdown
cannot be read are left alone.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..domain.circuits import circuit_code_blocks, gate_list
from ..ports import CircuitExporter, DocumentStore, StoredDocument

__all__ = ["QasmExportReport", "QasmExportService", "QasmFile"]

_UMASK = os.umask(0)
os.umask(_UMASK)  # read once at import, as the filesystem store does; os.umask is process-global


@dataclass(frozen=True)
class QasmFile:
    """One program written to disk and where it came from."""

    doc_id: str
    source_name: str
    block: int
    circuit: int
    part: int
    sign: str | None
    path: Path
    n_qubits: int
    n_clbits: int


@dataclass(frozen=True)
class QasmExportReport:
    """What one export wrote, removed, refused and warned about."""

    out: Path
    documents: int
    circuits: int
    files: tuple[QasmFile, ...]
    refusals: tuple[str, ...]
    warnings: tuple[str, ...]
    removed: int = 0  # stale programs of the exported documents deleted by this run

    @property
    def ok(self) -> bool:
        return not self.refusals

    @property
    def written(self) -> int:
        return len(self.files)


class QasmExportService:
    """Read stored documents, ask the exporter port, write one file per program."""

    def __init__(self, store: DocumentStore, out: Path, exporter: CircuitExporter):
        self.store = store
        self.out = Path(out)
        self.exporter = exporter

    def check_values(self, values: Mapping[str, str] | None = None) -> None:
        """Refuse a bad ``--set`` once, before a command exports any circuit."""
        self.exporter.check_values(values)

    def export_corpus(
        self,
        selectors: Sequence[str] = (),
        *,
        values: Mapping[str, str] | None = None,
        out: Path | None = None,
    ) -> QasmExportReport:
        """Export the selected documents (all of them when no selector is given)."""
        documents, warnings = self.store.corpus()
        if not documents:
            warnings = [*warnings, "no stored documents: ingest a paper first"]
        selected = _select(documents, selectors)
        return self._export(selected, values=values, out=out, warnings=warnings)

    def export_documents(
        self,
        documents: Sequence[StoredDocument],
        *,
        values: Mapping[str, str] | None = None,
        out: Path | None = None,
    ) -> QasmExportReport:
        """Export documents a caller already holds, as the post-ingest hook does.

        A document is exported once however often it is given (identical files share a doc_id),
        and as the corpus holds it: a partial or forced-OCR variant (``ingest --ocr-all``) gives
        way to the canonical document, whose programs share its file names.
        """
        chosen: dict[str, StoredDocument] = {}
        for document in documents:
            chosen.setdefault(document.manifest.doc_id, document)
        if any(not document.canonical for document in chosen.values()):
            corpus = {document.manifest.doc_id: document for document in self.store.corpus()[0]}
            chosen = {doc_id: corpus.get(doc_id, doc) for doc_id, doc in chosen.items()}
        return self._export(list(chosen.values()), values=values, out=out)

    def _export(
        self,
        documents: Sequence[StoredDocument],
        *,
        values: Mapping[str, str] | None,
        out: Path | None,
        warnings: Sequence[str] = (),
    ) -> QasmExportReport:
        self.exporter.check_values(values)
        target = Path(out or self.out)
        files: list[QasmFile] = []
        refusals: list[str] = []
        notes = list(warnings)
        unmatched = set(values or {})
        circuits = 0
        removed = 0
        for document in documents:
            manifest = document.manifest
            name = manifest.source_name
            try:
                markdown = self.store.markdown(document)
            except OSError as e:
                notes.append(f"{name}: cannot read the stored Markdown ({e})")
                continue
            found = 0
            written: set[Path] = set()
            for block_no, block in enumerate(circuit_code_blocks(markdown), 1):
                listing = gate_list(block.source)
                for circuit_no, circuit in enumerate(listing.circuits, 1):
                    found += 1
                    circuits += 1
                    export = self.exporter.export(
                        circuit,
                        values=values,
                        origin=f"{name}, block {block_no}, circuit {circuit_no}",
                        tool="docingest",
                    )
                    unmatched &= set(export.unmatched)
                    where = f"{name}, block {block_no}, circuit {circuit_no}"
                    for note in export.notes:
                        notes.append(f"{where}: {note}")
                    for program in export.programs:
                        filename = (
                            f"{manifest.doc_id[:16]}-b{block_no:02d}-"
                            f"c{circuit_no:02d}-p{program.part:02d}.qasm"
                        )
                        path = target / filename
                        _write(path, program.text)
                        written.add(path)
                        files.append(
                            QasmFile(
                                doc_id=manifest.doc_id,
                                source_name=name,
                                block=block_no,
                                circuit=circuit_no,
                                part=program.part,
                                sign=program.sign,
                                path=path,
                                n_qubits=program.n_qubits,
                                n_clbits=program.n_clbits,
                            )
                        )
                    for reason in export.reasons:
                        refusals.append(f"{where}: {reason}")
            stale = _remove_stale(target, manifest.doc_id, keep=written)
            removed += stale
            if stale:
                notes.append(f"{name}: removed {stale} stale QASM file(s) from an earlier export")
            if found == 0:
                notes.append(f"{name}: no circuits found")
        for key in sorted(unmatched):
            notes.append(f"binding {key!r} matches no symbol of any exported circuit")
        return QasmExportReport(
            out=target,
            documents=len(documents),
            circuits=circuits,
            files=tuple(files),
            refusals=tuple(refusals),
            warnings=tuple(notes),
            removed=removed,
        )


def _select(documents: Sequence[StoredDocument], selectors: Sequence[str]) -> list[StoredDocument]:
    """Documents named by doc_id, source name or a unique doc_id prefix of at least 8 chars."""
    if not selectors:
        return list(documents)
    chosen: dict[str, StoredDocument] = {}
    for selector in selectors:
        matches = [
            document
            for document in documents
            if selector in {document.manifest.doc_id, document.manifest.source_name}
            or (len(selector) >= 8 and document.manifest.doc_id.startswith(selector))
        ]
        if not matches:
            raise ValueError(f"no stored document matches {selector!r}")
        if len(matches) > 1:
            raise ValueError(
                f"{selector!r} matches {len(matches)} stored documents; use a longer doc_id prefix"
            )
        chosen[matches[0].manifest.doc_id] = matches[0]
    return list(chosen.values())


def _remove_stale(target: Path, doc_id: str, *, keep: set[Path]) -> int:
    """Delete this document's programs that this run did not write; return how many.

    Only ``<doc_id[:16]>-bNN-cNN-pNN.qasm`` files are touched: the programs of other documents,
    temporary files and anything else in the folder stay."""
    if not target.is_dir():
        return 0
    pattern = re.compile(rf"{re.escape(doc_id[:16])}-b\d+-c\d+-p\d+\.qasm")
    removed = 0
    for path in sorted(target.iterdir()):
        if path not in keep and pattern.fullmatch(path.name) and path.is_file():
            path.unlink()
            removed += 1
    return removed


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o666 & ~_UMASK)  # mkstemp creates 0600; match a normal write
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
