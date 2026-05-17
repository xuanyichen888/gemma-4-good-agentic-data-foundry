from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_data_foundry.database import build_sqlite_from_csv
from agentic_data_foundry.query import answer_question

EXPECTED_ROW_COUNT = 10
EXPECTED_COLUMN_COUNT = 9


def main() -> None:
    csv_path = ROOT / "examples" / "community_intake.csv"
    questions_path = ROOT / "examples" / "evaluation_questions.csv"

    # Use TemporaryDirectory so the .sqlite file is not held open on Windows
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "smoke_test.sqlite"

        result = build_sqlite_from_csv(csv_path, db_path, table_name="community_intake")
        assert result["row_count"] == EXPECTED_ROW_COUNT, (
            f"Expected {EXPECTED_ROW_COUNT} rows, got {result['row_count']}"
        )
        assert len(result["columns"]) == EXPECTED_COLUMN_COUNT, (
            f"Expected {EXPECTED_COLUMN_COUNT} columns, got {len(result['columns'])}"
        )

        with questions_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            eval_rows = list(reader)

        failures: list[str] = []
        for row in eval_rows:
            question = row["question"]
            # Use the explicit expected_sql_keywords column when present
            raw_keywords = row.get("expected_sql_keywords", row.get("expected_signal", ""))
            expected_keywords = [kw.strip() for kw in raw_keywords.split(",") if kw.strip()]
            try:
                answer = answer_question(db_path, result["table_name"], question)
                assert answer.sql.lower().startswith("select"), (
                    f"SQL did not start with SELECT: {answer.sql[:60]}"
                )
                assert isinstance(answer.rows, list), "rows must be a list"
                sql_lower = answer.sql.lower()
                for keyword in expected_keywords:
                    if keyword.lower() not in sql_lower:
                        failures.append(
                            f"  Q: {question!r}\n"
                            f"    Expected SQL keyword {keyword!r}\n"
                            f"    Got: {answer.sql.strip()}"
                        )
            except Exception as exc:
                failures.append(f"  Q: {question!r}\n    Error: {exc}")

    if failures:
        print(f"smoke test FAILED for {len(failures)} question(s):")
        for msg in failures:
            print(msg)
        sys.exit(1)

    print(f"smoke test passed for {len(eval_rows)} evaluation questions")



if __name__ == "__main__":
    main()
