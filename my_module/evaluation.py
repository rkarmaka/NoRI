import numpy as np
from scipy.spatial.distance import directed_hausdorff

def measure_confusion_matrix(true_label, predicted_label):
    """
    Computes the confusion matrix for binary classification.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.

    Returns:
    Tuple of (tp, fp, tn, fn), where:
    - tp: True positives
    - fp: False positives
    - tn: True negatives
    - fn: False negatives
    """
    if true_label.shape != predicted_label.shape:
        raise ValueError("Input images must have the same shape.")

    # Compare images pixel-wise
    tp = np.sum(np.logical_and(true_label, predicted_label))
    fp = np.sum(np.logical_and(true_label, np.logical_not(predicted_label)))
    tn = np.sum(np.logical_and(np.logical_not(true_label), np.logical_not(predicted_label)))
    fn = np.sum(np.logical_and(np.logical_not(true_label), predicted_label))

    return tp, fp, tn, fn


def measure_precision(true_label, predicted_label):
    """
    Computes precision for binary classification.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.

    Returns:
    Precision value.
    """
    tp, fp, _, _ = measure_confusion_matrix(true_label=true_label,
                                            predicted_label=predicted_label)

    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    return precision


def measure_recall(true_label, predicted_label):
    """
    Computes recall (sensitivity) for binary classification.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.

    Returns:
    Recall value.
    """
    tp, _, _, fn = measure_confusion_matrix(true_label=true_label,
                                            predicted_label=predicted_label)

    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    return recall


def measure_specificity(true_label, predicted_label):
    """
    Computes specificity for binary classification.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.

    Returns:
    Specificity value.
    """
    _, fp, tn, _ = measure_confusion_matrix(true_label=true_label,
                                            predicted_label=predicted_label)

    if (tn + fp) != 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    return specificity


def measure_accuracy(true_label, predicted_label):
    """
    Computes accuracy for binary classification.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.

    Returns:
    Accuracy value.
    """
    tp, fp, tn, fn = measure_confusion_matrix(true_label=true_label,
                                              predicted_label=predicted_label)

    acc = (tp + tn) / (tp + fp + tn + fn)

    return acc


def measure_fscore(true_label, predicted_label, beta=1):
    """
    Computes F-score for binary classification.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.
    - beta (float): Weight of precision in harmonic mean.

    Returns:
    F-score value.
    """
    precision = measure_precision(true_label=true_label,
                                  predicted_label=predicted_label)
    recall = measure_recall(true_label=true_label,
                            predicted_label=predicted_label)

    f_beta = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)

    return f_beta


def measure_MCC(true_label, predicted_label):
    """
    Computes Matthews correlation coefficient (MCC) for binary classification.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.

    Returns:
    MCC value.
    """
    tp, fp, tn, fn = measure_confusion_matrix(true_label=true_label,
                                              predicted_label=predicted_label)

    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc = num / den if den != 0 else 0.0

    return mcc


def measure_auc(true_label, predicted_label):
    """
    Computes Area Under the ROC Curve (AUC) for binary classification.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.

    Returns:
    AUC value.
    """
    tp, fp, tn, fn = measure_confusion_matrix(true_label=true_label,
                                              predicted_label=predicted_label)

    auc = 1 - 0.5 * ((fp / (fp + tn)) + (fn / (fn + tp)))

    return auc


def measure_kappa(true_label, predicted_label):
    """
    Computes Cohen's Kappa coefficient for binary classification.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.

    Returns:
    Kappa value.
    """
    tp, fp, tn, fn = measure_confusion_matrix(true_label=true_label,
                                              predicted_label=predicted_label)

    fc = (((tn + fn) * (tn + fp)) + ((fp + tp) * (fn + tp))) / (tp + tn + fn + fp)
    kappa = (tp + tn - fc) / (tp + tn + fp + fn - fc) if (tp + tn + fp + fn - fc) != 0 else 0.0

    return kappa


def measure_hausdorff(true_label, predicted_label, max=True):
    """
    Computes the Hausdorff distance between two binary images.

    Parameters:
    - true_label (numpy.ndarray): Ground truth labels.
    - predicted_label (numpy.ndarray): Predicted labels.
    - max (bool): If True, returns the maximum Hausdorff distance, else returns the mean.

    Returns:
    Hausdorff distance value.
    """
    # Compute Hausdorff distances in both directions
    distance_1to2 = directed_hausdorff(true_label, predicted_label)[0]
    distance_2to1 = directed_hausdorff(predicted_label, true_label)[0]

    if max:
        return np.max([distance_1to2, distance_2to1])
    else:
        return np.mean([distance_1to2, distance_2to1])
