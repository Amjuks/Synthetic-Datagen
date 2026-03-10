from __future__ import annotations

import pandas as pd

from tagging.taggers import RuleBasedTagger, TagDefinitions
from utils.io_utils import load_json, safe_write_csv


def tag_dataset(input_csv: str, output_csv: str, defs_path: str) -> pd.DataFrame:
    defs_json = load_json(defs_path)
    defs = TagDefinitions(**defs_json)
    tagger = RuleBasedTagger(defs)

    df = pd.read_csv(input_csv, engine="python")
    tags = [tagger.tag_row(row) for row in df.to_dict(orient="records")]
    tags_df = pd.DataFrame(tags)
    out_df = pd.concat([df, tags_df], axis=1)
    safe_write_csv(out_df, output_csv)
    return out_df
