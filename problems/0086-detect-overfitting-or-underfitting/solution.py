def model_fit_quality(training_accuracy, test_accuracy):
    """
    Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
    :param training_accuracy: float, training accuracy of the model (0 <= training_accuracy <= 1)
    :param test_accuracy: float, test accuracy of the model (0 <= test_accuracy <= 1)
    :return: int, one of 1, -1, or 0.
    """
    # Your code here
    from enum import Enum

    class Fits(Enum):
        OVERFITTING = 1
        UNDERFITTING = -1
        GOOD_FIT = 0
    
    diff = training_accuracy - test_accuracy
    if diff>0.2:
        return Fits.OVERFITTING.value
    elif training_accuracy<0.7 and test_accuracy<0.7:
        return Fits.UNDERFITTING.value
    else:
        return Fits.GOOD_FIT.value
