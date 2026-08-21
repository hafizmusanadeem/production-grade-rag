#!/usr/bin/env python3
"""
Production-grade batch processor.

Usage:
    python batch_processor.py /path/to/documents/
    python batch_processor.py --output report.json /path/to/documents/
    python batch_processor.py --resume report.json
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import logfire

from app.config import settings, validate_env_vars
from app.ingestion.processor import IngestionProcessor
from app.observability import configure_logfire


@dataclass
class ProcessingStats:
    """Statistics for a single file processing attempt."""

    file_name: str
    source_path: str
    file_size_kb: float
    status: str  # success, skipped, error
    chunks_created: Optional[int] = None
    error_message: Optional[str] = None
    processing_time_sec: Optional[float] = None
    local_path: Optional[str] = None


@dataclass
class BatchProcessingReport:
    """Summary report for an entire batch."""

    started_at: str
    completed_at: str
    source_directory: str
    total_files: int
    successful: int
    failed: int
    skipped: int
    total_chunks: int
    total_processing_time_sec: float
    data_directory: str
    qdrant_collection: str
    files: list[ProcessingStats]


class BatchProcessor:
    """Process multiple documents with checkpoint-aware recovery."""

    SUPPORTED_EXTENSIONS = frozenset(
        {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"}
    )

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or "DATA")
        self.processor = IngestionProcessor(data_dir=self.data_dir)

    def process_directory(
        self,
        directory: Path,
        resume_from: Optional[Path] = None,
    ) -> BatchProcessingReport:
        """Process all supported files in a directory."""

        directory = directory.resolve()

        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        files = sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ),
            key=lambda path: str(path).lower(),
        )

        if not files:
            raise ValueError(
                f"No supported files found in {directory}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        previous_report = self._load_resume_report(resume_from)

        processed_files = self._get_successful_paths(previous_report)

        embedding_files = {
            path.stem.removesuffix(".embeddings")
            for path in self.processor.embeddings_dir.glob("*.embeddings.json")
        }

        for file_path in files:
            if file_path.stem in embedding_files:
                processed_files.add(str(file_path.resolve()))

        logfire.info(
            "Local embedding checkpoints detected",
            embedding_file_count=len(embedding_files),
            skipped_file_count=len(processed_files),
        )

        start_time = datetime.now(UTC)
        current_stats: list[ProcessingStats] = []
        total_chunks = 0

        logfire.info(
            "Batch processing started",
            directory=str(directory),
            file_count=len(files),
            resume_enabled=resume_from is not None,
        )

        for index, file_path in enumerate(files, start=1):
            resolved_path = str(file_path.resolve())

            if resolved_path in processed_files:
                logfire.info(
                    "File skipped",
                    current=index,
                    total=len(files),
                    file_name=file_path.name,
                    reason="already processed or embedded",
                )
                continue

            file_stat = self._process_single_file(
                file_path=file_path,
                current=index,
                total=len(files),
            )

            current_stats.append(file_stat)

            if file_stat.status == "success":
                total_chunks += file_stat.chunks_created or 0

                logfire.info(
                    "File processed successfully",
                    current=index,
                    total=len(files),
                    file_name=file_path.name,
                    chunks_created=file_stat.chunks_created or 0,
                    processing_time_sec=file_stat.processing_time_sec,
                )

            elif file_stat.status == "skipped":
                logfire.warning(
                    "File skipped",
                    current=index,
                    total=len(files),
                    file_name=file_path.name,
                    reason=file_stat.error_message,
                )

            else:
                logfire.error(
                    "File processing failed",
                    current=index,
                    total=len(files),
                    file_name=file_path.name,
                    error=file_stat.error_message,
                )

        end_time = datetime.now(UTC)

        previous_stats = self._get_previous_stats(previous_report)

        stats = self._merge_stats(
            previous_stats=previous_stats,
            current_stats=current_stats,
        )

        successful = sum(stat.status == "success" for stat in stats)
        failed = sum(stat.status == "error" for stat in stats)
        skipped = sum(stat.status == "skipped" for stat in stats)

        total_chunks = sum(
            stat.chunks_created or 0
            for stat in stats
            if stat.status == "success"
        )

        report = BatchProcessingReport(
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat(),
            source_directory=str(directory),
            total_files=len(files),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_chunks=total_chunks,
            total_processing_time_sec=(end_time - start_time).total_seconds(),
            data_directory=str(self.data_dir),
            qdrant_collection=settings.QDRANT_COLLECTION or "unknown",
            files=stats,
        )

        self._log_summary(report)

        return report

    def _process_single_file(
        self,
        file_path: Path,
        current: int,
        total: int,
    ) -> ProcessingStats:
        """Process one file and convert failures into structured statistics."""

        start = datetime.now(UTC)
        resolved_path = file_path.resolve()

        try:
            file_size_kb = file_path.stat().st_size / 1024
        except OSError as exc:
            logfire.error(
                "Unable to inspect file",
                file_path=str(resolved_path),
                error=str(exc),
            )

            return ProcessingStats(
                file_name=file_path.name,
                source_path=str(resolved_path),
                file_size_kb=0.0,
                status="error",
                error_message=str(exc),
                processing_time_sec=(
                    datetime.now(UTC) - start
                ).total_seconds(),
            )

        try:
            logfire.info(
                "Processing file",
                current=current,
                total=total,
                file_name=file_path.name,
                file_size_kb=round(file_size_kb, 2),
            )

            result = self.processor.process(file_path)

            processing_time = (
                datetime.now(UTC) - start
            ).total_seconds()

            return ProcessingStats(
                file_name=file_path.name,
                source_path=str(resolved_path),
                file_size_kb=file_size_kb,
                status="success",
                chunks_created=result.chunk_count,
                processing_time_sec=processing_time,
                local_path=result.local_path,
            )

        except (ValueError, FileNotFoundError) as exc:
            processing_time = (
                datetime.now(UTC) - start
            ).total_seconds()

            logfire.warning(
                "File skipped due to expected processing error",
                file_path=str(resolved_path),
                error=str(exc),
            )

            return ProcessingStats(
                file_name=file_path.name,
                source_path=str(resolved_path),
                file_size_kb=file_size_kb,
                status="skipped",
                error_message=str(exc),
                processing_time_sec=processing_time,
            )

        except Exception as exc:
            processing_time = (
                datetime.now(UTC) - start
            ).total_seconds()

            logfire.exception(
                "Unexpected batch processing error",
                file_path=str(resolved_path),
            )

            return ProcessingStats(
                file_name=file_path.name,
                source_path=str(resolved_path),
                file_size_kb=file_size_kb,
                status="error",
                error_message=f"{type(exc).__name__}: {exc}",
                processing_time_sec=processing_time,
            )

    @staticmethod
    def _load_resume_report(
        resume_from: Optional[Path],
    ) -> Optional[dict]:
        """Load a previous report when resume mode is enabled."""

        if resume_from is None:
            return None

        resume_from = resume_from.resolve()

        if not resume_from.exists():
            raise ValueError(f"Resume report not found: {resume_from}")

        try:
            with resume_from.open("r", encoding="utf-8") as file:
                report = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON resume report: {resume_from}"
            ) from exc

        if not isinstance(report, dict) or "files" not in report:
            raise ValueError(
                f"Invalid resume report format: {resume_from}"
            )

        logfire.info(
            "Resume report loaded",
            report_path=str(resume_from),
            previous_file_count=len(report.get("files", [])),
        )

        return report

    @staticmethod
    def _get_successful_paths(
        report: Optional[dict],
    ) -> set[str]:
        """Return source paths successfully processed previously."""

        if not report:
            return set()

        return {
            str(Path(item["source_path"]).resolve())
            for item in report.get("files", [])
            if item.get("status") == "success"
            and item.get("source_path")
        }

    @staticmethod
    def _get_previous_stats(
        report: Optional[dict],
    ) -> list[ProcessingStats]:
        """Convert valid previous report entries into ProcessingStats."""

        if not report:
            return []

        stats: list[ProcessingStats] = []

        for item in report.get("files", []):
            try:
                stats.append(ProcessingStats(**item))
            except TypeError:
                logfire.warning(
                    "Invalid previous processing entry ignored",
                    entry=item,
                )

        return stats

    @staticmethod
    def _merge_stats(
        previous_stats: list[ProcessingStats],
        current_stats: list[ProcessingStats],
    ) -> list[ProcessingStats]:
        """
        Merge current results with previous results.

        Current results win when the same source path appears in both.
        This prevents duplicate entries during resume.
        """

        merged: dict[str, ProcessingStats] = {}

        for stat in previous_stats:
            merged[stat.source_path] = stat

        for stat in current_stats:
            merged[stat.source_path] = stat

        return sorted(
            merged.values(),
            key=lambda stat: stat.source_path.lower(),
        )

    @staticmethod
    def _log_summary(report: BatchProcessingReport) -> None:
        """Write structured batch summary to Logfire."""

        errors = [
            {
                "file_name": stat.file_name,
                "source_path": stat.source_path,
                "error": stat.error_message,
            }
            for stat in report.files
            if stat.status == "error"
        ]

        logfire.info(
            "Batch processing completed",
            total_files=report.total_files,
            successful=report.successful,
            skipped=report.skipped,
            failed=report.failed,
            total_chunks=report.total_chunks,
            total_processing_time_sec=report.total_processing_time_sec,
            average_chunks_per_successful_file=(
                report.total_chunks / report.successful
                if report.successful
                else 0.0
            ),
            data_directory=report.data_directory,
            qdrant_collection=report.qdrant_collection,
            error_count=len(errors),
            errors=errors,
        )


def save_report(
    report: BatchProcessingReport,
    output_path: Path,
) -> None:
    """Persist a processing report as JSON."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(report)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    logfire.info(
        "Batch processing report saved",
        output_path=str(output_path),
    )


def load_and_continue(
    report_path: Path,
    output_path: Optional[Path] = None,
) -> None:
    """Resume processing using the source directory stored in a report."""

    report_path = report_path.resolve()

    if not report_path.exists():
        raise ValueError(f"Report not found: {report_path}")

    try:
        with report_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON report: {report_path}"
        ) from exc

    required_fields = {
        "source_directory",
        "data_directory",
        "files",
    }

    missing_fields = required_fields - data.keys()

    if missing_fields:
        raise ValueError(
            "Resume report is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    directory = Path(data["source_directory"])
    processor = BatchProcessor(data_dir=data["data_directory"])

    new_report = processor.process_directory(
        directory=directory,
        resume_from=report_path,
    )

    save_report(
        new_report,
        output_path or report_path,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Production-grade batch document processor."
    )

    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Directory containing documents to process.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path where the JSON processing report should be saved.",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a previous JSON processing report.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("DATA"),
        help="Data directory used for local processing artifacts.",
    )

    return parser


def main() -> int:
    """CLI entry point."""

    configure_logfire()

    try:
        validate_env_vars()
    except Exception:
        logfire.exception("Environment validation failed")
        return 1

    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        if args.resume:
            load_and_continue(
                report_path=args.resume,
                output_path=args.output,
            )
            return 0

        if not args.directory:
            parser.error(
                "directory is required unless --resume is provided"
            )

        processor = BatchProcessor(data_dir=str(args.data_dir))

        report = processor.process_directory(
            directory=args.directory,
        )

        if args.output:
            save_report(report, args.output)

        return 0 if report.failed == 0 else 1

    except ValueError as exc:
        logfire.error(
            "Batch processor validation error",
            error=str(exc),
        )
        return 1

    except Exception:
        logfire.exception("Batch processor failed unexpectedly")
        return 1


if __name__ == "__main__":
    sys.exit(main())