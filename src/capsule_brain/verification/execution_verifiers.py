"""Execution-backed verifiers.

These verifiers use ``ExecutionService`` to actually run code — compile checks,
pytest, ruff, mypy — rather than relying on static analysis alone. This bridges
the gap between verification and execution, making Capsule Brain a
verifier-driven coding agent rather than a system with two disconnected
facilities.

All execution-backed verifiers are SKIP by default unless the relevant
``content_type`` is set in metadata, so they don't interfere with non-code
verification.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from .models import VerificationCheck, VerificationStatus
from .verifiers import Verifier


class ExecutionVerifier(Verifier):
    """Base class for verifiers that run code via ExecutionService.

    Subclasses define ``name``, ``content_type``, ``command_template``, and
    ``file_extension``. The subject (code/artifact) is written to a temp file
    and the command is executed against it.
    """

    name = "execution_base"
    content_type: str | None = None
    file_extension: str = ".txt"
    command_template: list[str] = []

    def __init__(self, execution_service: Any) -> None:
        self.execution = execution_service

    async def verify(
        self,
        subject: str,
        metadata: dict[str, Any],
    ) -> VerificationCheck:
        if metadata.get("content_type") != self.content_type:
            return VerificationCheck(
                self.name,
                VerificationStatus.SKIP,
                f"Not {self.content_type}.",
            )

        from capsule_brain.execution.models import ExecutionRequest

        sandbox_root = Path(self.execution.policy.cwd_root).resolve()
        sandbox_root.mkdir(parents=True, exist_ok=True)
        artifact: Path | None = None
        try:
            # Unique files prevent concurrent verifier runs from overwriting or
            # deleting each other's artifacts. Keep the file inside cwd_root so
            # both host and container execution policies accept it.
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=self.file_extension,
                prefix=f"verify_{self.name}_",
                delete=False,
                dir=sandbox_root,
                encoding="utf-8",
            ) as handle:
                handle.write(subject)
                artifact = Path(handle.name)

            # Commands run with cwd=sandbox_root; use a relative artifact name
            # so the same request works when cwd is mounted as /workspace in a
            # container.
            request = ExecutionRequest(
                command=self._build_command(artifact.name),
                cwd=str(sandbox_root),
                source=f"verifier:{self.name}",
                metadata={"verifier": self.name},
            )
            result = await self.execution.execute(request)
            return self._interpret(result)
        finally:
            if artifact is not None:
                try:
                    os.unlink(artifact)
                except OSError:
                    pass

    def _build_command(self, artifact_path: str) -> list[str]:
        """Build the command, replacing ``{artifact}`` with the file path."""
        return [
            arg.replace("{artifact}", artifact_path)
            for arg in self.command_template
        ]

    def _interpret(self, result: Any) -> VerificationCheck:
        """Convert an ExecutionResult into a VerificationCheck."""
        if result.timed_out:
            return VerificationCheck(
                self.name,
                VerificationStatus.FAIL,
                f"{self.name} timed out after {result.duration_ms:.0f}ms.",
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timed_out": True,
                },
            )
        if result.exit_code == 0:
            return VerificationCheck(
                self.name,
                VerificationStatus.PASS,
                f"{self.name} passed.",
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "duration_ms": result.duration_ms,
                },
            )
        return VerificationCheck(
            self.name,
            VerificationStatus.FAIL,
            f"{self.name} failed with exit code {result.exit_code}.",
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
            },
        )


class PythonCompileVerifier(ExecutionVerifier):
    """Verify Python code compiles without syntax errors."""

    name = "python_compile"
    content_type = "python"
    file_extension = ".py"
    # -B suppresses .pyc writes (workspace may be read-only in container).
    # py_compile returns non-zero on syntax errors; compileall does not.
    command_template = ["python", "-B", "-m", "py_compile", "{artifact}"]


class PytestVerifier(ExecutionVerifier):
    """Run pytest against the subject (treated as a test file)."""

    name = "pytest"
    content_type = "pytest"
    file_extension = ".py"
    # PYTHONDONTWRITEBYTECODE is set by the container runner; for host
    # execution, pytest's -p no:cacheprovider avoids cache writes.
    command_template = ["pytest", "-v", "-p", "no:cacheprovider", "{artifact}"]


class RuffVerifier(ExecutionVerifier):
    """Run ruff linter against the subject."""

    name = "ruff"
    content_type = "ruff"
    file_extension = ".py"
    command_template = ["ruff", "check", "{artifact}"]


class MypyVerifier(ExecutionVerifier):
    """Run mypy type checker against the subject."""

    name = "mypy"
    content_type = "mypy"
    file_extension = ".py"
    command_template = ["mypy", "{artifact}"]


def create_execution_verifiers(
    execution_service: Any,
    *,
    include: set[str] | None = None,
) -> list[ExecutionVerifier]:
    """Create execution-backed verifiers for an ExecutionService.

    By default creates all available verifiers. Pass ``include`` to select a
    subset by name (e.g. ``{"python_compile", "pytest"}``).
    """
    all_verifiers = [
        PythonCompileVerifier(execution_service),
        PytestVerifier(execution_service),
        RuffVerifier(execution_service),
        MypyVerifier(execution_service),
    ]
    if include is None:
        return all_verifiers
    return [v for v in all_verifiers if v.name in include]
