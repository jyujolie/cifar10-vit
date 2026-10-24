from src.data import stratified_split_indices


def test_stratified_split_is_reproducible_and_has_no_overlap():
    labels = [0] * 10 + [1] * 10
    train_a, val_a = stratified_split_indices(labels, val_per_class=3, seed=42)
    train_b, val_b = stratified_split_indices(labels, val_per_class=3, seed=42)

    assert (train_a, val_a) == (train_b, val_b)
    assert len(val_a) == 6
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a + val_a) == set(range(20))


def test_stratified_split_balances_every_class():
    labels = [0] * 10 + [1] * 10 + [2] * 10
    _, val = stratified_split_indices(labels, val_per_class=4, seed=7)

    assert len(val) == 12
    for label in (0, 1, 2):
        assert sum(1 for index in val if labels[index] == label) == 4


def test_stratified_split_rejects_invalid_sizes():
    labels = [0] * 10 + [1] * 2
    for invalid in (0, -1, 10, 2):
        try:
            stratified_split_indices(labels, val_per_class=invalid, seed=42)
        except ValueError:
            pass
        else:
            raise AssertionError(f"val_per_class={invalid} should be rejected")
