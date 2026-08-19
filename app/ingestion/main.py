#!/usr/bin/env python3
"""
Main entry point for RAG data ingestion pipeline.

Usage:
    # Process a single file
    python main.py process /path/to/document.pdf

    # Process all files in a directory (progress, resume, and reporting
    # handled by batch_processor.py under the hood)
    python main.py process-dir /path/to/documents/

    # Process with a specific data directory
    python main.py process /path/to/document.pdf --data-dir ./DATA
"""

import sys
from pathlib import Path
from typing import Optional

import logfire
from app.config import settings
from app.observability import configure_logfire
from app.ingestion.processor import process_file


def configure_logging() -> None:
    """Configure logfire observability."""
    configure_logfire()


def process_single_file(file_path: str, data_dir: Optional[str] = None) -> None:
    """
    Process a single file through the ingestion pipeline.

    Args:
        file_path: Path to the file to process
        data_dir: Optional custom data directory for embeddings storage
    """
    path = Path(file_path)

    if not path.exists():
        print(f"❌ Error: File not found: {path}")
        sys.exit(1)

    print(f"🚀 Processing: {path.name}")
    print(f"   File type: {path.suffix}")
    print(f"   Size: {path.stat().st_size / 1024:.2f} KB")

    try:
        result = process_file(file_path, data_dir=data_dir)

        print(f"\n✅ Processing Complete!")
        print(f"   Chunks created: {result.chunk_count}")
        print(f"   Local storage: {result.local_path}")
        print(f"   Qdrant collection: {result.qdrant_collection}")
        print(f"   Points upserted: {result.qdrant_points_upserted}")

    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logfire.error("Processing failed", error=str(e), file_path=str(path))
        sys.exit(1)


def process_directory(dir_path: str, data_dir: Optional[str] = None) -> None:
    """
    Process all supported files in a directory.

    Delegates to batch_processor.BatchProcessor so directory processing has
    exactly one implementation (progress tracking, per-file stats, resume
    support) instead of two slightly-different copies of the same logic.

    Supported formats: .pdf, .docx, .pptx, .html, .htm, .txt, .md
    """
    from batch_processor import BatchProcessor

    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        print(f"❌ Error: Directory not found: {directory}")
        sys.exit(1)

    try:
        processor = BatchProcessor(data_dir=data_dir)
        processor.process_directory(directory)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logfire.error("Directory processing failed", error=str(e), directory=str(directory))
        sys.exit(1)


def validate_environment() -> None:
    """Validate required environment variables."""
    required = ["GEMINI_API_KEY", "QDRANT_API_KEY", "QDRANT_CLUSTER_ENDPOINT", "QDRANT_COLLECTION"]
    missing = [var for var in required if not getattr(settings, var)]

    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\n📋 Please set these in your .env file")
        sys.exit(1)


def print_usage() -> None:
    """Print usage information."""
    print("""
Usage: python main.py <command> <path> [options]

Commands:
    process <file_path>       Process a single file
    process-dir <dir_path>    Process all files in a directory (with progress + resume)
    help                      Show this help message

Options:
    --data-dir DIR           Custom directory for embeddings storage (default: ./DATA)

Examples:
    python main.py process document.pdf
    python main.py process-dir ./documents/
    python main.py process data.docx --data-dir ./embeddings

For directory batches with a saved JSON report or --resume support, use
batch_processor.py directly:
    python batch_processor.py --output report.json ./documents/
    python batch_processor.py --resume report.json

Supported file types:
    .pdf, .docx, .pptx, .html, .htm, .txt, .md
    """)


def main() -> None:
    """Main entry point."""
    configure_logging()
    validate_environment()

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "help" or command in ["-h", "--help"]:
        print_usage()
        sys.exit(0)

    if command == "process":
        if len(sys.argv) < 3:
            print("Error: 'process' command requires a file path")
            print_usage()
            sys.exit(1)
        file_path = sys.argv[2]
        data_dir = None
        if len(sys.argv) > 3 and sys.argv[3] == "--data-dir":
            data_dir = sys.argv[4] if len(sys.argv) > 4 else None
        process_single_file(file_path, data_dir=data_dir)

    elif command == "process-dir":
        if len(sys.argv) < 3:
            print("Error: 'process-dir' command requires a directory path")
            print_usage()
            sys.exit(1)
        dir_path = sys.argv[2]
        data_dir = None
        if len(sys.argv) > 3 and sys.argv[3] == "--data-dir":
            data_dir = sys.argv[4] if len(sys.argv) > 4 else None
        process_directory(dir_path, data_dir=data_dir)

    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()