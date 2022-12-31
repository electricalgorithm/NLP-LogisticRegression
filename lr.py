import sys
import numpy as np


class ErrorWriter:
    """This class is a writer for the error files."""

    def __init__(self, file_to_write) -> None:
        self.file = file_to_write

    def write(
        self, train_error: float, test_error: float, validation_error: float = None
    ) -> None:
        """Writes given errors to the file."""
        string_to_write = (
            f"error(train): {train_error:.6f}\nerror(test): {test_error:.6f}"
        )
        with open(self.file, "w", encoding="utf-8") as output_file:
            output_file.write(string_to_write)


class FeatureReader:
    """This class is a reader for the feature files."""

    def __init__(self, feature_file) -> None:
        self.file = np.loadtxt(feature_file, delimiter="\t")

    @property
    def all(self) -> np.ndarray:
        """Returns the feature."""
        return self.file

    @property
    def features(self) -> np.ndarray:
        """Returns the features."""
        return self.file[:, 1:]

    @property
    def labels(self) -> np.ndarray:
        """Returns the labels."""
        return self.file[:, 0]


class BinaryLogisticRegression:
    """Logistic Regression model."""

    def __init__(self) -> None:
        # Internal attributes.
        self._weights: np.ndarray = None
        self._is_trained: bool = False
        self._num_epoch: int = None
        self._learning_rate: float = None

    def train(
        self,
        input_matrix: np.ndarray,
        label_vector: np.ndarray,
        num_epoch: int,
        learning_rate: int,
        add_intercept: bool = True,
    ) -> tuple:
        """
        Trains the model using the given input vectors and labels,
        it performs the stochastic gradient descent algorithm.
        """
        # Copy the input matrix to avoid modifying the original one.
        c_input_matrix = input_matrix.copy()

        # Add a bias term to the input matrix.
        if add_intercept:
            c_input_matrix = np.insert(c_input_matrix, 0, 1, axis=1)

        # Initialize the weight vector with zeros.
        if self._weights is None:
            self._weights = np.zeros(c_input_matrix.shape[1])

        # Do the training for epoches.
        for _ in range(num_epoch):
            predictions = self.predict(
                c_input_matrix, weights=self._weights, add_bias=False
            )
            prediction_difference_from_labels = predictions - label_vector
            gradient = np.dot(prediction_difference_from_labels.T, c_input_matrix)
            self._weights = self._weights - learning_rate * gradient

        self._is_trained = True
        return (self.compute_error(predictions, label_vector), predictions)

    def predict(
        self,
        input_matrix: np.ndarray,
        weights: np.ndarray = None,
        add_bias: bool = True,
    ) -> np.ndarray:
        """Predicts the labels for the given input vectors."""
        # Check if the model is trained.
        if not self._is_trained and weights is None:
            raise SystemExit("[ERROR] Model is not trained yet.")

        # Use the trained weights if no weights are provided.
        if weights is None:
            weights = self._weights

        # Copy the input matrix to avoid modifying the original one.
        c_input_matrix = input_matrix.copy()

        # Add a bias term to the input matrix.
        if add_bias:
            c_input_matrix = np.insert(c_input_matrix, 0, 1, axis=1)

        resulting_vector = np.dot(c_input_matrix, weights)
        sigmoid_of_results = self.sigmoid(resulting_vector)
        return np.where(sigmoid_of_results >= 0.5, 1, 0)

    def calculate_cross_entropy_error(
        self, predicted_labels: np.ndarray, real_labels: np.ndarray
    ) -> float:
        """Calculates the cross entropy error for the given input vectors and labels."""
        cross_entropy = np.sum(
            np.where(
                real_labels == 1,
                np.log(predicted_labels),
                np.log(1 - predicted_labels),
            )
        )
        return -cross_entropy

    @staticmethod
    def compute_error(predicted_labels: np.ndarray, real_labels: np.ndarray) -> float:
        """Computes the error between the predicted and the actual labels."""
        return np.mean(np.abs(predicted_labels - real_labels))

    @staticmethod
    def sigmoid(unknown: np.ndarray) -> np.ndarray:
        """Implementation of the sigmoid function."""
        euler = np.exp(unknown)
        return euler / (1 + euler)


class LGCalculator:
    """Main functionality of the program."""

    def __init__(self) -> None:
        raise SystemExit("[ERROR] This class is not meant to be instantiated.")

    @staticmethod
    def main():
        """Main function of the program."""
        settings = LGCalculator.argument_parser()

        # Read the datasets prepared with feature.py script.
        train_dataset = FeatureReader(settings["train"]["input_file"])
        test_dataset = FeatureReader(settings["test"]["input_file"])

        # Train the model.
        blg = BinaryLogisticRegression()
        train_error, train_predictions = blg.train(
            train_dataset.features,
            train_dataset.labels,
            num_epoch=int(settings["model_options"]["num_epoch"]),
            learning_rate=float(settings["model_options"]["learning_rate"]),
        )

        # Predict the labels for the test set.
        test_predictions = blg.predict(test_dataset.features)
        test_error = blg.compute_error(test_predictions, test_dataset.labels)

        # Write the metrics to the output file.
        ErrorWriter(settings["model_options"]["metrics_output"]).write(
            train_error, test_error
        )

        # Write the predictions to the output file.
        np.savetxt(
            fname=settings["train"]["output_file"],
            X=train_predictions,
            fmt="%d",
            newline="\n",
        )
        np.savetxt(
            fname=settings["test"]["output_file"],
            X=test_predictions,
            fmt="%d",
            newline="\n",
        )

    @staticmethod
    def argument_parser() -> dict:
        """It parses the arguments given."""
        args = sys.argv[1:]
        if len(args) != 8:
            raise SystemExit("[ERROR] Invalid number of arguments.")

        return {
            "model_options": {
                "metrics_output": args[5],
                "num_epoch": args[6],
                "learning_rate": args[7],
            },
            "train": {
                "input_file": args[0],
                "output_file": args[3],
            },
            "test": {
                "input_file": args[2],
                "output_file": args[4],
            },
            "validation": {
                "input_file": args[1],
            },
        }


if __name__ == "__main__":
    LGCalculator.main()

