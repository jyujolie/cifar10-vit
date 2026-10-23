from src.data import split_indices


def test_split_indices_is_reproducible_and_has_no_overlap():
    train_a, val_a = split_indices(total=20, val_size=4, seed=42)
    train_b, val_b = split_indices(total=20, val_size=4, seed=42)

    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 16
    assert len(val_a) == 4
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a + val_a) == set(range(20))


def test_split_indices_rejects_empty_partition():
    for invalid_size in (0, 10):
        try:
            split_indices(total=10, val_size=invalid_size, seed=42)
        except ValueError:
            pass
        else:
            raise AssertionError(f"val_size={invalid_size} should be rejected")
