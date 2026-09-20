from vector_index.data import chunk_text


def test_chunk_text_sizes():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text(text, 100)
    assert [len(c.split()) for c in chunks] == [100, 100, 50]


def test_chunk_text_drops_tiny_tail():
    text = " ".join(f"w{i}" for i in range(105))
    assert [len(c.split()) for c in chunk_text(text, 100, min_words=20)] == [100]


def test_chunk_text_handles_whitespace_and_empty():
    assert chunk_text("   \n\t ", 10) == []
    assert chunk_text("a  b\n\nc", 10, min_words=1) == ["a b c"]
