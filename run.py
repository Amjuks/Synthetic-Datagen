from __future__ import annotations

import argparse
from pathlib import Path

from deduplication.deduplicate import deduplicate_dataset
from generator.generate_dataset import DatasetGenerator, DatasetRunConfig
from generator.model_loader import ModelLoader
from reporting.report_generator import generate_report
from tagging.tag_dataset import tag_dataset
from utils.gpu_detect import detect_hardware, hardware_to_dict
from utils.io_utils import load_json
from utils.logging_utils import setup_logger

LIGHTWEIGHT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Domain-specialized synthetic dataset generator")
    p.add_argument("--config", required=True, help="Path to JSON config template")
    p.add_argument("--samples", type=int, required=True, help="Number of samples to generate")
    p.add_argument(
        "--output-name",
        help="Base output name (without extension). Produces <name>.csv, <name>_tagged.csv, <name>_deduplicated.csv",
    )
    p.add_argument(
        "--turn-mode",
        choices=["single", "multi"],
        help="Single turn uses one instruction/response. Multi turn uses instruction_n/response_n columns.",
    )
    p.add_argument("--min-turns", type=int, help="Minimum turns for multi-turn generation (>=2)")
    p.add_argument("--max-turns", type=int, help="Maximum turns for multi-turn generation (>= min-turns)")
    p.add_argument("--seed", type=int, help="Optional random seed (omit for different output every run)")
    p.add_argument("--reasoning", dest="reasoning", action="store_true", help="Enable reasoning column(s)")
    p.add_argument("--no-reasoning", dest="reasoning", action="store_false", help="Disable reasoning column(s)")
    p.set_defaults(reasoning=None)
    p.add_argument("--output-dir", default="datasets", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    config = load_json(args.config)
    config.setdefault("dataset_structure", {})
    config.setdefault("generation", {})
    config["model_name"] = LIGHTWEIGHT_MODEL

    if args.turn_mode:
        config["dataset_structure"]["multi_turn"] = args.turn_mode == "multi"
    if args.min_turns is not None:
        config["dataset_structure"]["min_turns"] = args.min_turns
    if args.max_turns is not None:
        config["dataset_structure"]["max_turns"] = args.max_turns
    if args.reasoning is not None:
        config["dataset_structure"]["include_reasoning"] = args.reasoning
    if args.seed is not None:
        config["generation"]["seed"] = args.seed

    multi_turn = bool(config["dataset_structure"].get("multi_turn", False))
    if multi_turn:
        min_turns = int(config["dataset_structure"].get("min_turns", 2))
        max_turns = int(config["dataset_structure"].get("max_turns", min_turns))
        if min_turns < 2:
            raise ValueError("For multi-turn mode, --min-turns must be at least 2.")
        if max_turns < min_turns:
            raise ValueError("--max-turns must be greater than or equal to --min-turns.")
    else:
        config["dataset_structure"]["min_turns"] = 1
        config["dataset_structure"]["max_turns"] = 1

    logger.info(
        "Run options: model=%s multi_turn=%s turns=%s-%s reasoning=%s seed=%s samples=%s",
        LIGHTWEIGHT_MODEL,
        multi_turn,
        config["dataset_structure"].get("min_turns", 1),
        config["dataset_structure"].get("max_turns", 1),
        bool(config["dataset_structure"].get("include_reasoning", False)),
        config["generation"].get("seed"),
        args.samples,
    )

    hw = detect_hardware()
    logger.info("Hardware detected: %s", hardware_to_dict(hw))

    if bool(config.get("generation", {}).get("template_only", False)):
        raise ValueError(
            "template_only mode is disabled. Set generation.template_only=false and choose a real model_name."
        )

    model_name = LIGHTWEIGHT_MODEL
    model_loader = ModelLoader(model_name=model_name, device=hw.device)
    backend = model_loader.load()
    logger.info("Using backend=%s model=%s", backend.backend_name, model_name)
    model_generate_fn = backend.generator

    run_cfg = DatasetRunConfig(
        dataset_type=config["dataset_type"],
        system_prompt=config["system_prompt"],
        domain_structure=config.get("domain_structure", {}),
        dataset_structure=config.get("dataset_structure", {}),
        generation=config.get("generation", {}),
        samples=args.samples,
        random_seed=config.get("generation", {}).get("seed"),
    )

    out_dir = Path(args.output_dir)
    if args.output_name:
        rel_base = Path(args.output_name)
        if rel_base.suffix:
            rel_base = rel_base.with_suffix("")
    else:
        rel_base = Path(f"{config['dataset_type']}_dataset")
    base_path = out_dir / rel_base
    base_name = rel_base.as_posix()
    raw_csv = str(base_path.with_suffix(".csv"))
    tagged_csv = str(base_path.parent / f"{base_path.name}_tagged.csv")
    dedup_csv = str(base_path.parent / f"{base_path.name}_deduplicated.csv")
    report_json = str(base_path.parent / f"{base_path.name}_report.json")
    report_md = str(base_path.parent / f"{base_path.name}_report.md")

    generator = DatasetGenerator(model_generate_fn, logger)
    df = generator.run(run_cfg, raw_csv)
    logger.info("Generated %s rows -> %s", len(df), raw_csv)

    tagged_df = tag_dataset(raw_csv, tagged_csv, "tagging/tag_definitions.json")
    logger.info("Tagged dataset rows=%s -> %s", len(tagged_df), tagged_csv)

    threshold = float(config.get("deduplication", {}).get("similarity_threshold", 0.92))
    dedup_df = deduplicate_dataset(tagged_csv, dedup_csv, similarity_threshold=threshold)
    logger.info("Deduplicated dataset: %s -> %s rows", len(tagged_df), len(dedup_df))

    report = generate_report(
        base_name=base_name,
        config_path=args.config,
        samples_requested=args.samples,
        raw_df=df,
        tagged_df=tagged_df,
        dedup_df=dedup_df,
        raw_csv=raw_csv,
        tagged_csv=tagged_csv,
        dedup_csv=dedup_csv,
        report_json=report_json,
        report_md=report_md,
    )
    logger.info(
        "Report generated: %s (json) and %s (md). Dedup removed=%s rows (%s%%).",
        report_json,
        report_md,
        report["pipeline"]["dedup_removed"],
        report["pipeline"]["dedup_removed_pct"],
    )


if __name__ == "__main__":
    main()
