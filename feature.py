import sys
import csv
import numpy as np

VECTOR_LEN = 300  # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(
        file, delimiter="\t", comments=None, encoding="utf-8", dtype="l,O"
    )
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter="\t")
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


def argument_parser() -> dict:
    """
    It parses the arguments given.
    :return: The argument dictionary.
    """
    args = sys.argv[1:]
    if len(args) != 7:
        raise SystemExit("[ERROR] Invalid number of arguments.")

    return {
        "word2vec_file": args[3],
        "train": {
            "input_file": args[0],
            "output_file": args[4],
        },
        "test": {
            "input_file": args[2],
            "output_file": args[6],
        },
        "validation": {
            "input_file": args[1],
            "output_file": args[5],
        },
    }

def mean_sum_of_the_review_word2vec_vectors(clean_dataset, word2vec_map):
    """Desc
    """
    dataset_to_return = []
    for label, review in clean_dataset:
        weights_combined = np.zeros(VECTOR_LEN)
        for word in review.split():
            weights_combined += word2vec_map[word]
        dataset_to_return.append((label, weights_combined / len(review.split())))
    return dataset_to_return


def clean_non_word2vec_words(dataset, word2vec_map):
    """Desc
    """
    cleaned_dataset = []
    for label, review in dataset:
        cleaned_review = []
        for word in review.split():
            if word in word2vec_map:
                cleaned_review.append(word)
        cleaned_dataset.append((label, " ".join(cleaned_review)))
    return cleaned_dataset


if __name__ == "__main__":
    settings = argument_parser()
    word2vec_map = load_feature_dictionary(settings["word2vec_file"])
    train_dataset = load_tsv_dataset(settings["train"]["input_file"])
    test_dataset = load_tsv_dataset(settings["test"]["input_file"])
    validation_dataset = load_tsv_dataset(settings["validation"]["input_file"])

    # Clean operation.
    train_dataset = clean_non_word2vec_words(train_dataset, word2vec_map)
    test_dataset = clean_non_word2vec_words(test_dataset, word2vec_map)
    validation_dataset = clean_non_word2vec_words(validation_dataset, word2vec_map)

    # Feature extraction.
    processed_train = mean_sum_of_the_review_word2vec_vectors(train_dataset, word2vec_map)
    processed_test = mean_sum_of_the_review_word2vec_vectors(test_dataset, word2vec_map)
    processed_validation = mean_sum_of_the_review_word2vec_vectors(validation_dataset, word2vec_map)

    # Save the cleaned datasets.
    with open(settings["train"]["output_file"], "w", encoding="utf-8") as f:
        for label, review in processed_train:
            f.write("%.6f\t" % float(label) + "\t".join(["%.6f" % feature for feature in review]))
            f.write("\n")

    with open(settings["test"]["output_file"], "w", encoding="utf-8") as f:
        for label, review in processed_test:
            f.write("%.6f\t" % float(label) + "\t".join(["%.6f" % feature for feature in review]))
            f.write("\n")

    with open(settings["validation"]["output_file"], "w", encoding="utf-8") as f:
        for label, review in processed_validation:
            f.write("%.6f\t" % float(label) + "\t".join(["%.6f" % feature for feature in review]))
            f.write("\n")
