def compute_accuracy(pred_y, test_y):
    return (pred_y == test_y).sum() / len(pred_y)