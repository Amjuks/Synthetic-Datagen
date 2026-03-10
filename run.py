from __future__ import annotations

import argparse
from pathlib import Path

from deduplication.deduplicate import deduplicate_dataset
from generator.generate_dataset import DatasetGenerator, DatasetRunConfig
from generator.model_loader import ModelLoader
from tagging.tag_dataset import tag_dataset
from utils.gpu_detect import detect_hardware, hardware_to_dict
from utils.io_utils import load_json
from utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Domain-specialized synthetic dataset generator")
    p.add_argument("--config", required=True, help="Path to JSON config template")
    p.add_argument("--samples", type=int, required=True, help="Number of samples to generate")
    p.add_argument("--output-dir", default="datasets", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    config = load_json(args.config)
    hw = detect_hardware()
    logger.info("Hardware detected: %s", hardware_to_dict(hw))

    model_name = config.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
    model_loader = ModelLoader(model_name=model_name, device=hw.device)
    backend = model_loader.load()
    logger.info("Using backend=%s model=%s", backend.backend_name, model_name)

    run_cfg = DatasetRunConfig(
        dataset_type=config["dataset_type"],
        system_prompt=config["system_prompt"],
        domain_structure=config["domain_structure"],
        dataset_structure=config.get("dataset_structure", {}),
        generation=config.get("generation", {}),
        samples=args.samples,
    )

    out_dir = Path(args.output_dir)
    base_name = f"{config['dataset_type']}_dataset"
    raw_csv = str(out_dir / f"{base_name}.csv")
    tagged_csv = str(out_dir / f"{base_name}_tagged.csv")
    dedup_csv = str(out_dir / f"{base_name}_deduplicated.csv")

    generator = DatasetGenerator(backend.generator, logger)
    df = generator.run(run_cfg, raw_csv)
    logger.info("Generated %s rows -> %s", len(df), raw_csv)

    tagged_df = tag_dataset(raw_csv, tagged_csv, "tagging/tag_definitions.json")
    logger.info("Tagged dataset rows=%s -> %s", len(tagged_df), tagged_csv)

    threshold = float(config.get("deduplication", {}).get("similarity_threshold", 0.92))
    dedup_df = deduplicate_dataset(tagged_csv, dedup_csv, similarity_threshold=threshold)
    logger.info("Deduplicated dataset: %s -> %s rows", len(tagged_df), len(dedup_df))


if __name__ == "__main__":
    main()
