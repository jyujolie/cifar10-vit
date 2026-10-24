from torchvision import transforms
from torchvision.transforms import RandAugment

from src.data import image_transform, stratified_split_indices


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


def has_operation(pipeline, operation_type):
    return any(isinstance(step, operation_type) for step in pipeline.transforms)


def test_rand_augment_is_optional_and_only_in_training():
    plain = image_transform(training=True)
    strong = image_transform(training=True, rand_augment_ops=2, rand_augment_magnitude=9)

    assert not has_operation(plain, RandAugment)
    assert has_operation(strong, RandAugment)
    assert strong.transforms[2].num_ops == 2
    assert strong.transforms[2].magnitude == 9


def test_evaluation_transform_never_augments():
    for options in ({}, {"rand_augment_ops": 2, "rand_augment_magnitude": 9}):
        pipeline = image_transform(training=False, **options)
        assert not has_operation(pipeline, RandAugment)
        assert not has_operation(pipeline, transforms.RandomCrop)
        assert not has_operation(pipeline, transforms.RandomHorizontalFlip)


def test_rand_augment_arguments_must_be_paired():
    for ops, magnitude in ((2, 0), (0, 9), (-1, 9), (2, -1)):
        try:
            image_transform(
                training=True, rand_augment_ops=ops, rand_augment_magnitude=magnitude
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"ops={ops}, magnitude={magnitude} should be rejected")
