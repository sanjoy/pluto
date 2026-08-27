#!/usr/bin/env python3
"""Regenerates the tiny FineWeb-schema Parquet fixture beside this script."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    table = pa.table(
        {
            "text": ["Hello, world!", "The quick brown fox.", "naïve café 🌍"],
            "id": ["id-1", "id-2", "id-3"],
            "dump": ["CC-MAIN-2026-30"] * 3,
            "url": [
                "https://example.test/one",
                "https://example.test/two",
                "https://example.test/three",
            ],
            "file_path": [
                "s3://example/one.warc.gz",
                "s3://example/two.warc.gz",
                "s3://example/three.warc.gz",
            ],
            "language": ["en"] * 3,
            "language_score": pa.array([0.99, 0.95, 0.91], type=pa.float64()),
            "token_count": pa.array([4, 5, 7], type=pa.int64()),
            "score": pa.array([4.0, 3.5, 3.0], type=pa.float64()),
            "int_score": pa.array([4, 4, 3], type=pa.int64()),
        }
    )
    pq.write_table(
        table,
        Path(__file__).with_name("fineweb_sample.parquet"),
        row_group_size=2,
        version="2.6",
        data_page_version="1.0",
        compression="snappy",
        use_dictionary=True,
        write_page_index=False,
    )


if __name__ == "__main__":
    main()
