import numpy as np

def learning_curve(
    X_train, y_train, X_val, y_val, train_sizes, degree,
    bias_threshold=0.5, variance_threshold=0.5
):
    X_train = np.asarray(X_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    X_trains = []
    X_vals = []

    for i in range(degree + 1):
        X_trains.append(X_train ** i)
        X_vals.append(X_val ** i)

    X_train_new = np.hstack(X_trains)
    X_val_new = np.hstack(X_vals)

    train_errors = []
    val_errors = []

    for size in train_sizes:
        X_subset = X_train_new[:size]
        y_subset = y_train[:size]

        # Fit on this training prefix using the pseudoinverse
        theta = np.linalg.pinv(X_subset) @ y_subset

        training_predictions = X_subset @ theta
        val_predictions = X_val_new @ theta

        train_mse = np.mean((training_predictions - y_subset) ** 2)
        val_mse = np.mean((val_predictions - y_val) ** 2)

        train_errors.append(float(train_mse))
        val_errors.append(float(val_mse))

    # The problem specifies the last entries
    final_train = train_errors[-1]
    final_val = val_errors[-1]

    if final_train > bias_threshold:
        diagnosis = 'high_bias'
    elif final_val - final_train > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'

    return {
        'train_errors': train_errors,
        'val_errors': val_errors,
        'diagnosis': diagnosis
    }