#!/usr/bin/env python3
"""
Batch processor with progress tracking, error recovery, and detailed reporting.

Usage:
    python batch_processor.py process-dir /path/to/documents/
    python batch_processor.py --resume /path/to/processed.json
    python batch_processor.py --output report.json /path/to/documents/
"""

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import logfire
from app.config import settings
from app.ingestion.processor import IngestionProcessor, process_file


@dataclass
class ProcessingStats:
    """Statistics for a single file processing."""
    file_name: str
    file_size_kb: float
    status: str  # 'success', 'skipped', 'error'
    chunks_created: Optional[int] = None
    error_message: Optional[str] = None
    processing_time_sec: Optional[float] = None
    local_path: Optional[str] = None


@dataclass
class BatchProcessingReport:
    """Summary report for entire batch processing."""
    started_at: str
    completed_at: str
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
    """Process multiple documents with progress tracking."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or "DATA")
        self.processor = IngestionProcessor(data_dir=self.data_dir)
        self.supported_extensions = {
            ".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"
        }
    
    def process_directory(
        self,
        directory: Path,
        resume_from: Optional[Path] = None,
    ) -> BatchProcessingReport:
        """
        Process all supported files in directory.
        
        Args:
            directory: Path to directory containing files
            resume_from: Optional path to previous processing report for resuming
        
        Returns:
            BatchProcessingReport with complete statistics
        """
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        
        # Get all supported files
        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in self.supported_extensions
        ]
        
        if not files:
            raise ValueError(
                f"No supported files found in {directory}. "
                f"Supported: {', '.join(self.supported_extensions)}"
            )
        
        # Load previous results if resuming
        processed_files = set()
        if resume_from and resume_from.exists():
            with open(resume_from) as f:
                prev_report = json.load(f)
                processed_files = {
                    s["file_name"] for s in prev_report["files"]
                    if s["status"] == "success"
                }
                print(f"🔄 Resuming from {len(processed_files)} previously processed files")
        
        start_time = datetime.now(UTC)
        stats = []
        total_chunks = 0
        
        print(f"📁 Processing {len(files)} file(s) from {directory}")
        print("=" * 70)
        
        for idx, file_path in enumerate(files, 1):
            if file_path.name in processed_files:
                print(f"[{idx}/{len(files)}] ⏭️  {file_path.name} (already processed)")
                continue
            
            file_stat = self._process_single_file(file_path, idx, len(files))
            stats.append(file_stat)
            
            if file_stat.status == "success":
                total_chunks += file_stat.chunks_created or 0
                print(
                    f"[{idx}/{len(files)}] ✅ {file_path.name} "
                    f"({file_stat.chunks_created} chunks, {file_stat.processing_time_sec:.2f}s)"
                )
            elif file_stat.status == "skipped":
                print(f"[{idx}/{len(files)}] ⏭️  {file_path.name} (skipped: {file_stat.error_message})")
            else:
                print(f"[{idx}/{len(files)}] ❌ {file_path.name} (error: {file_stat.error_message})")
        
        end_time = datetime.now(UTC)
        total_time = (end_time - start_time).total_seconds()
        
        # Generate report
        successful = len([s for s in stats if s.status == "success"])
        failed = len([s for s in stats if s.status == "error"])
        skipped = len([s for s in stats if s.status == "skipped"])
        
        report = BatchProcessingReport(
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat(),
            total_files=len(files),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_chunks=total_chunks,
            total_processing_time_sec=total_time,
            data_directory=str(self.data_dir),
            qdrant_collection=settings.QDRANT_COLLECTION or "unknown",
            files=stats,
        )
        
        self._print_summary(report)
        return report
    
    def _process_single_file(
        self,
        file_path: Path,
        current: int,
        total: int,
    ) -> ProcessingStats:
        """Process a single file with error handling."""
        start = datetime.now(UTC)
        file_size_kb = file_path.stat().st_size / 1024
        
        try:
            result = self.processor.process(file_path)
            processing_time = (datetime.now(UTC) - start).total_seconds()
            
            return ProcessingStats(
                file_name=file_path.name,
                file_size_kb=file_size_kb,
                status="success",
                chunks_created=result.chunk_count,
                processing_time_sec=processing_time,
                local_path=result.local_path,
            )
        
        except (ValueError, FileNotFoundError) as e:
            # Expected errors - skip silently
            return ProcessingStats(
                file_name=file_path.name,
                file_size_kb=file_size_kb,
                status="skipped",
                error_message=str(e),
            )
        
        except Exception as e:
            # Unexpected errors - log and continue
            processing_time = (datetime.now(UTC) - start).total_seconds()
            logfire.error(
                "Batch processing error",
                file_path=str(file_path),
                error=str(e),
            )
            
            return ProcessingStats(
                file_name=file_path.name,
                file_size_kb=file_size_kb,
                status="error",
                error_message=str(e),
                processing_time_sec=processing_time,
            )
    
    def _print_summary(self, report: BatchProcessingReport) -> None:
        """Print formatted summary."""
        print("\n" + "=" * 70)
        print("📊 BATCH PROCESSING REPORT")
        print("=" * 70)
        print(f"\n⏱️  Processing Time: {report.total_processing_time_sec:.2f}s")
        print(f"📁 Files Processed: {report.total_files}")
        print(f"   ✅ Successful: {report.successful}")
        print(f"   ⚠️  Skipped: {report.skipped}")
        print(f"   ❌ Failed: {report.failed}")
        print(f"\n📈 Total Chunks Created: {report.total_chunks}")
        
        if report.total_chunks > 0:
            avg_chunks_per_file = report.total_chunks / report.successful
            print(f"   Average per file: {avg_chunks_per_file:.1f}")
        
        print(f"\n💾 Storage:")
        print(f"   Local: {report.data_directory}")
        print(f"   Qdrant Collection: {report.qdrant_collection}")
        print(f"\n⏱️  Started: {report.started_at}")
        print(f"   Completed: {report.completed_at}")
        
        # Show errors if any
        errors = [f for f in report.files if f.status == "error"]
        if errors:
            print(f"\n⚠️  {len(errors)} file(s) with errors:")
            for file_stat in errors:
                print(f"   - {file_stat.file_name}: {file_stat.error_message}")


def save_report(report: BatchProcessingReport, output_path: Path) -> None:
    """Save processing report to JSON file."""
    data = asdict(report)
    # Convert ProcessingStats to dicts
    data["files"] = [asdict(f) for f in report.files]
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📄 Report saved: {output_path}")


def load_and_continue(report_path: Path) -> None:
    """Continue processing from a previous report."""
    if not report_path.exists():
        print(f"❌ Report not found: {report_path}")
        sys.exit(1)
    
    with open(report_path) as f:
        data = json.load(f)
    
    directory = Path(data["data_directory"]).parent
    processor = BatchProcessor(data_dir=data["data_directory"])
    
    new_report = processor.process_directory(directory, resume_from=report_path)
    
    # Save updated report with timestamp
    output_path = report_path.parent / f"report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    save_report(new_report, output_path)


def print_usage() -> None:
    """Print usage information."""
    print("""
Usage: python batch_processor.py [OPTIONS] <directory>

Options:
    --output FILE           Save report to JSON file
    --resume FILE           Resume from previous processing report
    --data-dir DIR         Custom directory for embeddings (default: ./DATA)

Examples:
    # Basic batch processing
    python batch_processor.py ./documents/
    
    # With report output
    python batch_processor.py --output report.json ./documents/
    
    # Resume previous run
    python batch_processor.py --resume report.json ./documents/
    
    # Custom data directory
    python batch_processor.py --data-dir ./embeddings ./documents/
    """)


def main() -> None:
    """Main entry point."""
    logfire.configure(
        token=settings.LOGFIRE_TOKEN,
        service_name=settings.LOGFIRE_SERVICE_NAME,
        environment=settings.ENVIRONMENT,
        send_to_logfire="if-token-present",
    )
    
    # Check required env vars
    if not settings.GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set")
        sys.exit(1)
    if not settings.QDRANT_API_KEY:
        print("❌ QDRANT_API_KEY not set")
        sys.exit(1)
    
    args = sys.argv[1:]
    
    if not args or "--help" in args or "-h" in args:
        print_usage()
        sys.exit(0)
    
    # Parse arguments
    output_file = None
    resume_file = None
    data_dir = None
    directory = None
    
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        elif args[i] == "--resume" and i + 1 < len(args):
            resume_file = args[i + 1]
            i += 2
        elif args[i] == "--data-dir" and i + 1 < len(args):
            data_dir = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            directory = args[i]
            i += 1
        else:
            i += 1
    
    # Handle resume mode
    if resume_file:
        load_and_continue(Path(resume_file))
        return
    
    if not directory:
        print("❌ Directory path required")
        print_usage()
        sys.exit(1)
    
    # Process directory
    try:
        processor = BatchProcessor(data_dir=data_dir)
        report = processor.process_directory(Path(directory))
        
        # Save report if requested
        if output_file:
            save_report(report, Path(output_file))
    
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logfire.error("Batch processing failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
