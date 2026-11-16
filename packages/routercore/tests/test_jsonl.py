from routercore.adapters.jsonl import read_jsonl


def test_reads_one_object_per_line_skipping_blank_lines_and_keeping_utf8(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"text": "cómo"}\n\n{"text": "ok"}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"text": "cómo"}, {"text": "ok"}]
