from src.evaluate import select_examples, summarize_predictions


def test_metrics_count_all_predictions_including_errors():
    metrics = summarize_predictions([0, 1, 1, 2], [0, 0, 1, 2], num_classes=3)
    assert metrics['accuracy'] == 0.75
    assert metrics['correct'] == 3
    assert metrics['samples'] == 4
    assert metrics['confusion_matrix'] == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]


def test_examples_are_separated_by_actual_prediction_result():
    correct, wrong = select_examples([0, 1, 1, 2], [0, 0, 1, 2], count=4)
    assert set(correct) == {0, 2, 3}
    assert wrong == [1]
