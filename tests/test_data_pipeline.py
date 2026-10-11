from training.data.make_mlx_jsonl import (
    chat_example,
    deduplicate,
    format_medqa,
    grouped_split,
    meddialog_rows,
)
from training.data.prepare_datasets import legacy_text_to_messages


def test_grouped_split_has_no_group_leakage():
    rows = [
        chat_example(
            f"question {index}",
            f"answer {index}",
            source="source-a",
            group_id=f"group-{index // 2}",
        )
        for index in range(20)
    ]

    train, valid = grouped_split(rows, validation_ratio=0.2, seed=42)
    train_groups = {row["group_id"] for row in train}
    valid_groups = {row["group_id"] for row in valid}

    assert train
    assert valid
    assert train_groups.isdisjoint(valid_groups)


def test_deduplicate_normalizes_case():
    first = chat_example("Headache", "Hydrate", source="a", group_id="1")
    duplicate = chat_example("headache", "hydrate", source="b", group_id="2")
    assert deduplicate([first, duplicate]) == [first]


def test_meddialog_rows_end_with_assistant():
    rows = meddialog_rows(
        {
            "description": "Adult with a mild respiratory complaint",
            "utterances": [
                "Patient: I have been coughing for two days.",
                "Doctor: Monitor your breathing and seek care if symptoms worsen.",
            ],
        },
        index=7,
    )
    assert len(rows) == 1
    assert [message["role"] for message in rows[0]["messages"]] == ["user", "assistant"]


def test_medqa_does_not_request_a_missing_rationale_or_train_one_word_answers():
    row = format_medqa(
        {
            "question": "Which treatment is most appropriate?",
            "answer": "Supportive care",
            "options": {"A": "Antibiotics", "B": "Supportive care"},
        },
        index=1,
    )
    assert row is not None
    assert "rationale" not in row["messages"][0]["content"].lower()
    assert len(row["messages"][1]["content"].split()) >= 8


def test_legacy_phi3_text_is_migrated_to_messages():
    messages = legacy_text_to_messages(
        "<|user|>\nWhat is anemia?<|end|>\n<|assistant|>\nA low red-cell state.<|end|>"
    )
    assert messages == [
        {"role": "user", "content": "What is anemia?"},
        {"role": "assistant", "content": "A low red-cell state."},
    ]
