"""
Central orchestrator for the NEET Counselling Dataset Pipeline.
Manages the full lifecycle: discovery -> download -> extraction -> normalization -> validation -> DB -> ML dataset.
"""
import asyncio
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
import structlog

from src.agents.link_discovery import MCCLinkDiscoveryAgent
from src.agents.state_link_discovery import StatePortalDiscoveryAgent
from src.agents.downloader import DownloadAgent
from src.agents.pdf_extractor import PDFExtractionAgent
from src.agents.ocr_agent import OCRAgent
from src.agents.normalizer import DataNormalizer
from src.agents.validator import ValidationAgent
from src.agents.db_ingestion import DBIngestionAgent
from src.agents.qa_agent import QADatasetBuilder
from src.utils.checkpoint import CheckpointManager


logger = structlog.get_logger(__name__)


class PipelineCoordinator:
    """
    Coordinates all pipeline phases with checkpoint/resume support.

    Phases:
        1: Link Discovery - crawl MCC archive, find PDF URLs
        2: Download - download all discovered PDFs
        3: PDF Extraction - extract tables from PDFs + OCR fallback
        4: Normalization - clean, deduplicate, normalize fields
        5: DB Ingestion - upsert into PostgreSQL
        6: Validation - validate data quality
        7: ML Dataset - build ML-ready train/test splits
    """

    PHASE_NAMES = {
        1: "Link Discovery",
        2: "Download",
        3: "PDF Extraction",
        4: "Data Normalization",
        5: "DB Ingestion",
        6: "Validation",
        7: "ML Dataset",
        8: "State Portal Discovery",
        9: "State Portal Download + Extract",
        10: "State Portal Normalize + Retrain",
    }

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.extracted_dir = self.data_dir / "extracted"
        self.cleaned_dir = self.data_dir / "cleaned"
        self.ml_ready_dir = self.data_dir / "ml_ready"
        self.checkpoint = CheckpointManager(str(self.data_dir / "checkpoints"))
        self.start_time: Optional[float] = None

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _log_phase_start(self, phase: int):
        name = self.PHASE_NAMES.get(phase, f"Phase {phase}")
        logger.info("phase_started", phase=phase, name=name)
        print(f"\n{'='*60}")
        print(f"  PHASE {phase}: {name}")
        print(f"{'='*60}")

    def _log_phase_complete(self, phase: int, duration: float, stats: dict = None):
        name = self.PHASE_NAMES.get(phase, f"Phase {phase}")
        logger.info(
            "phase_completed",
            phase=phase,
            name=name,
            duration_seconds=duration,
            stats=stats,
        )
        print(f"\n  Phase {phase} completed in {duration:.1f}s")
        if stats:
            for k, v in stats.items():
                print(f"    {k}: {v}")

    # -- Phase 1: Link Discovery ------------------------------------------

    async def run_phase_1(self) -> Path:
        self._log_phase_start(1)
        start = time.time()

        agent = MCCLinkDiscoveryAgent(output_dir=self.data_dir)
        inventory_path = await agent.discover(str(self.config_path))

        self._log_phase_complete(1, time.time() - start, {
            "output": str(inventory_path),
        })
        return inventory_path

    # -- Phase 2: Download ------------------------------------------------

    async def run_phase_2(self, inventory_path: Path) -> Path:
        self._log_phase_start(2)
        start = time.time()

        concurrency = self.config.get("downloader", {}).get("max_concurrent", 8)
        agent = DownloadAgent(output_dir=self.data_dir, concurrency=concurrency)
        manifest_path = await agent.download_all(inventory_path)

        self._log_phase_complete(2, time.time() - start, {
            "output": str(manifest_path),
        })
        return manifest_path

    # -- Phase 3: PDF Extraction ------------------------------------------

    async def run_phase_3(self, manifest_path: Path) -> tuple[Path, Path]:
        self._log_phase_start(3)
        start = time.time()

        extractor = PDFExtractionAgent(output_dir=self.data_dir)
        extracted_dir, extraction_log = await extractor.extract_all(manifest_path)

        ocr_agent = OCRAgent(output_dir=self.data_dir)

        confidence_threshold = self.config.get("extraction", {}).get(
            "confidence_threshold", 0.8
        )
        low_confidence_files: list[Path] = []
        if extraction_log.exists():
            with open(extraction_log) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        conf = float(row.get("confidence", 1.0))
                    except (ValueError, TypeError):
                        conf = 1.0
                    if conf < confidence_threshold:
                        local_path = row.get("output_path", "")
                        if local_path:
                            pdf_path = Path(local_path)
                            if pdf_path.exists():
                                low_confidence_files.append(pdf_path)

        if low_confidence_files:
            logger.info("ocr_needed", count=len(low_confidence_files))
            await ocr_agent.ocr_extract(low_confidence_files)

        self._log_phase_complete(3, time.time() - start, {
            "extracted_dir": str(extracted_dir),
            "ocr_files": len(low_confidence_files),
        })
        return extracted_dir, extraction_log

    # -- Phase 4: Normalization -------------------------------------------

    async def run_phase_4(self, extracted_dir: Path) -> Path:
        self._log_phase_start(4)
        start = time.time()

        normalizer = DataNormalizer(output_dir=self.data_dir)
        normalized_path = await normalizer.normalize(extracted_dir)

        self._log_phase_complete(4, time.time() - start, {
            "output": str(normalized_path),
        })
        return normalized_path

    # -- Phase 5: DB Ingestion --------------------------------------------

    async def run_phase_5(self, normalized_path: Path) -> dict:
        self._log_phase_start(5)
        start = time.time()

        db_config = self.config.get("database", {})
        db_url = (
            f"postgresql+asyncpg://{db_config.get('user', 'neet_user')}"
            f":{db_config.get('password', 'neet_pass_2024')}"
            f"@{db_config.get('host', 'localhost')}"
            f":{db_config.get('port', 5432)}"
            f"/{db_config.get('name', 'neet_counselling')}"
        )
        batch_size = self.config.get("db_ingestion", {}).get("batch_size", 1000)
        max_workers = self.config.get("db_ingestion", {}).get("max_concurrent", 3)

        ingester = DBIngestionAgent(
            db_url=db_url,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        stats = await ingester.ingest(normalized_path)

        self._log_phase_complete(5, time.time() - start, stats)
        return stats

    # -- Phase 6: Validation -----------------------------------------------

    async def run_phase_6(self, normalized_path: Path) -> tuple[Path, dict]:
        self._log_phase_start(6)
        start = time.time()

        max_workers = self.config.get("validation", {}).get("max_concurrent", 3)
        validator = ValidationAgent(max_concurrency=max_workers)
        report_path, stats = await validator.validate(normalized_path)

        self._log_phase_complete(6, time.time() - start, stats)
        return report_path, stats

    # -- Phase 7: ML Dataset -----------------------------------------------

    async def run_phase_7(self, normalized_path: Path) -> dict:
        self._log_phase_start(7)
        start = time.time()

        db_config = self.config.get("database", {})
        db_url = (
            f"postgresql+asyncpg://{db_config.get('user', 'neet_user')}"
            f":{db_config.get('password', 'neet_pass_2024')}"
            f"@{db_config.get('host', 'localhost')}"
            f":{db_config.get('port', 5432)}"
            f"/{db_config.get('name', 'neet_counselling')}"
        )

        builder = QADatasetBuilder(db_url=db_url)
        stats = await builder.build_ml_dataset(normalized_path)

        self._log_phase_complete(7, time.time() - start, stats)
        return stats

    # -- Phase 8: State Portal Discovery ------------------------------------

    async def run_phase_8(self) -> Path:
        self._log_phase_start(8)
        start = time.time()

        agent = StatePortalDiscoveryAgent(output_dir=self.data_dir)
        inventory_path = await agent.discover()

        self._log_phase_complete(8, time.time() - start, {
            "output": str(inventory_path),
        })
        return inventory_path

    # -- Phase 9: State Portal Download + Extract ---------------------------

    async def run_phase_9(self, state_inventory_path: Path) -> Path:
        self._log_phase_start(9)
        start = time.time()

        # Download state portal PDFs
        concurrency = self.config.get("downloader", {}).get("max_concurrent", 8)
        downloader = DownloadAgent(output_dir=self.data_dir, concurrency=concurrency)
        manifest_path = await downloader.download_all(state_inventory_path)

        # Extract tables
        extractor = PDFExtractionAgent(output_dir=self.data_dir)
        extracted_dir, extraction_log = await extractor.extract_all(manifest_path)

        self._log_phase_complete(9, time.time() - start, {
            "manifest": str(manifest_path),
            "extracted_dir": str(extracted_dir),
        })
        return extracted_dir

    # -- Phase 10: State Portal Normalize + Retrain -------------------------

    async def run_phase_10(self, state_extracted_dir: Path) -> dict:
        self._log_phase_start(10)
        start = time.time()

        # Normalize state data
        normalizer = DataNormalizer(output_dir=self.data_dir)
        state_normalized_path = await normalizer.normalize(state_extracted_dir)

        # Merge with existing MCC data
        mcc_path = self.data_dir / "cleaned" / "normalized_dataset.csv"
        state_path = self.data_dir / "cleaned" / "normalized_dataset.csv"

        # Re-run ML dataset build with combined data
        db_config = self.config.get("database", {})
        db_url = (
            f"postgresql+asyncpg://{db_config.get('user', 'neet_user')}"
            f":{db_config.get('password', 'neet_pass_2024')}"
            f"@{db_config.get('host', 'localhost')}"
            f":{db_config.get('port', 5432)}"
            f"/{db_config.get('name', 'neet_counselling')}"
        )
        builder = QADatasetBuilder(db_url=db_url)
        stats = await builder.build_ml_dataset(state_normalized_path)

        self._log_phase_complete(10, time.time() - start, stats)
        return stats

    # -- Full Pipeline -----------------------------------------------------

    async def run_full_pipeline(self):
        """Run all phases sequentially with checkpointing."""
        self.start_time = time.time()

        print(f"\n{'#'*60}")
        print(f"  NEET Counselling Dataset Pipeline")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")

        try:
            inventory_path = await self.run_phase_1()
            self.checkpoint.save_checkpoint("1", {"inventory_path": str(inventory_path)})

            manifest_path = await self.run_phase_2(inventory_path)
            self.checkpoint.save_checkpoint("2", {"manifest_path": str(manifest_path)})

            extracted_dir, extraction_log = await self.run_phase_3(manifest_path)
            self.checkpoint.save_checkpoint("3", {"extracted_dir": str(extracted_dir)})

            normalized_path = await self.run_phase_4(extracted_dir)
            self.checkpoint.save_checkpoint("4", {"normalized_path": str(normalized_path)})

            db_stats = await self.run_phase_5(normalized_path)
            self.checkpoint.save_checkpoint("5", db_stats)

            report_path, val_stats = await self.run_phase_6(normalized_path)
            self.checkpoint.save_checkpoint(
                "6", {"report_path": str(report_path), **val_stats}
            )

            ml_stats = await self.run_phase_7(normalized_path)
            self.checkpoint.save_checkpoint("7", ml_stats)

            total_duration = time.time() - self.start_time
            print(f"\n{'#'*60}")
            print(f"  PIPELINE COMPLETE")
            print(f"  Total time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
            print(f"{'#'*60}")

        except Exception as e:
            logger.error("pipeline_failed", error=str(e), exc_info=True)
            print(f"\n  PIPELINE FAILED: {e}")
            raise

    # -- Resume From Phase -------------------------------------------------

    async def run_from_phase(self, phase: int):
        """Resume pipeline from a specific phase."""
        self.start_time = time.time()

        print(f"\n  Resuming from Phase {phase}: {self.PHASE_NAMES.get(phase, '')}")

        normalized_path: Optional[Path] = None

        if phase == 1:
            inventory_path = await self.run_phase_1()
            self.checkpoint.save_checkpoint("1", {"inventory_path": str(inventory_path)})
            manifest_path = await self.run_phase_2(inventory_path)
            self.checkpoint.save_checkpoint("2", {"manifest_path": str(manifest_path)})
            extracted_dir, extraction_log = await self.run_phase_3(manifest_path)
            self.checkpoint.save_checkpoint("3", {"extracted_dir": str(extracted_dir)})
            normalized_path = await self.run_phase_4(extracted_dir)
            self.checkpoint.save_checkpoint("4", {"normalized_path": str(normalized_path)})
        elif phase == 2:
            cp = self.checkpoint.load_checkpoint("1")
            if not cp:
                raise RuntimeError("No checkpoint for phase 1. Run from start.")
            manifest_path = await self.run_phase_2(Path(cp["inventory_path"]))
            self.checkpoint.save_checkpoint("2", {"manifest_path": str(manifest_path)})
            extracted_dir, extraction_log = await self.run_phase_3(manifest_path)
            self.checkpoint.save_checkpoint("3", {"extracted_dir": str(extracted_dir)})
            normalized_path = await self.run_phase_4(extracted_dir)
            self.checkpoint.save_checkpoint("4", {"normalized_path": str(normalized_path)})
        elif phase == 3:
            cp = self.checkpoint.load_checkpoint("2")
            if not cp:
                raise RuntimeError("No checkpoint for phase 2. Run from start.")
            extracted_dir, extraction_log = await self.run_phase_3(
                Path(cp["manifest_path"])
            )
            self.checkpoint.save_checkpoint("3", {"extracted_dir": str(extracted_dir)})
            normalized_path = await self.run_phase_4(extracted_dir)
            self.checkpoint.save_checkpoint("4", {"normalized_path": str(normalized_path)})
        elif phase == 4:
            cp = self.checkpoint.load_checkpoint("3")
            if not cp:
                raise RuntimeError("No checkpoint for phase 3. Run from start.")
            normalized_path = await self.run_phase_4(Path(cp["extracted_dir"]))
            self.checkpoint.save_checkpoint("4", {"normalized_path": str(normalized_path)})
        elif phase in (5, 6, 7):
            cp = self.checkpoint.load_checkpoint("4")
            if not cp:
                raise RuntimeError("No checkpoint for phase 4. Run from start.")
            normalized_path = Path(cp["normalized_path"])
        elif phase in (8, 9, 10):
            normalized_path = None  # state phases are independent

        if 5 <= phase <= 7:
            if normalized_path is None:
                cp = self.checkpoint.load_checkpoint("4")
                if not cp:
                    raise RuntimeError("No checkpoint for phase 4. Run from start.")
                normalized_path = Path(cp["normalized_path"])
            if phase >= 5:
                await self.run_phase_5(normalized_path)
            if phase >= 6:
                await self.run_phase_6(normalized_path)
            if phase >= 7:
                await self.run_phase_7(normalized_path)
        elif phase >= 8:
            if phase == 8:
                state_inv = await self.run_phase_8()
                self.checkpoint.save_checkpoint("8", {"inventory_path": str(state_inv)})
            if phase == 9:
                cp8 = self.checkpoint.load_checkpoint("8")
                inv_path = Path(cp8["inventory_path"]) if cp8 else self.data_dir / "state_links_inventory.csv"
                state_ext = await self.run_phase_9(inv_path)
                self.checkpoint.save_checkpoint("9", {"extracted_dir": str(state_ext)})
            if phase == 10:
                cp9 = self.checkpoint.load_checkpoint("9")
                ext_dir = Path(cp9["extracted_dir"]) if cp9 else self.data_dir / "extracted"
                await self.run_phase_10(ext_dir)


def main():
    parser = argparse.ArgumentParser(description="NEET Counselling Dataset Pipeline")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to config file",
    )
    parser.add_argument("--phase", type=int, help="Run specific phase (1-10)")
    parser.add_argument("--resume", type=int, help="Resume from specific phase")
    parser.add_argument(
        "--status", action="store_true", help="Show pipeline status"
    )
    args = parser.parse_args()

    coordinator = PipelineCoordinator(config_path=args.config)

    if args.status:
        for phase_num in range(1, 11):
            status = coordinator.checkpoint.load_checkpoint(str(phase_num))
            phase_name = coordinator.PHASE_NAMES.get(phase_num, "")
            if status:
                print(f"  Phase {phase_num}: COMPLETED - {phase_name}")
            else:
                print(f"  Phase {phase_num}: PENDING   - {phase_name}")
        return

    if args.resume:
        asyncio.run(coordinator.run_from_phase(args.resume))
    elif args.phase:
        asyncio.run(coordinator.run_from_phase(args.phase))
    else:
        asyncio.run(coordinator.run_full_pipeline())


if __name__ == "__main__":
    main()
