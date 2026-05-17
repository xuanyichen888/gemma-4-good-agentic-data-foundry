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


def main() -> None:
    csv_path = ROOT / "examples" / "community_intake.csv"
    questions_path = ROOT / "examples" / "evaluation_questions.csv"

    with tempfile.NamedTemporaryFile(suffix=".sqlite") as db_file:
        result = build_sqlite_from_csv(csv_path, db_file.name, table_name="community_intake")
        assert result["row_count"] == 10
        assert len(result["columns"]) == 9

        with questions_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            questions = [row["question"] for row in reader]

        for question in questions:
            answer = answer_question(db_file.name, result["table_name"], question)
            assert answer.sql.lower().startswith("select")
            assert isinstance(answer.rows, list)

    print(f"smoke test passed for {len(questions)} evaluation questions")


if __name__ == "__main__":
    main()
