# CS5760 Natural Language Processing - Homework 2
# Student Name: Shaik Ameer Pasha
# Student ID: 700782176

from collections import Counter


def confusion_matrix_metrics(matrix, class_names):
    """Compute per-class, macro, and micro precision/recall."""
    n = len(class_names)
    precisions = []
    recalls = []

    print("Part I - Q5: Confusion Matrix Metrics")
    for i, name in enumerate(class_names):
        tp = matrix[i][i]
        predicted_total = sum(matrix[i])          # row total
        gold_total = sum(matrix[r][i] for r in range(n))  # column total

        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / gold_total if gold_total else 0.0
        precisions.append(precision)
        recalls.append(recall)

        print(f"{name}: Precision = {precision:.4f}, Recall = {recall:.4f}")

    macro_precision = sum(precisions) / n
    macro_recall = sum(recalls) / n

    total_tp = sum(matrix[i][i] for i in range(n))
    total_predictions = sum(sum(row) for row in matrix)
    total_gold = total_predictions

    micro_precision = total_tp / total_predictions
    micro_recall = total_tp / total_gold

    print(f"Macro Precision = {macro_precision:.4f}")
    print(f"Macro Recall = {macro_recall:.4f}")
    print(f"Micro Precision = {micro_precision:.4f}")
    print(f"Micro Recall = {micro_recall:.4f}")


def train_bigram_model(corpus):
    """Return unigram and bigram counts for a tokenized corpus."""
    unigram_counts = Counter()
    bigram_counts = Counter()

    for sentence in corpus:
        tokens = sentence.split()
        unigram_counts.update(tokens)
        bigram_counts.update(zip(tokens[:-1], tokens[1:]))

    return unigram_counts, bigram_counts


def bigram_probability(previous_word, word, unigram_counts, bigram_counts):
    """MLE bigram probability P(word | previous_word)."""
    denominator = unigram_counts[previous_word]
    if denominator == 0:
        return 0.0
    return bigram_counts[(previous_word, word)] / denominator


def sentence_probability(sentence, unigram_counts, bigram_counts):
    """Calculate a sentence probability using MLE bigram probabilities."""
    tokens = sentence.split()
    probability = 1.0

    for previous_word, word in zip(tokens[:-1], tokens[1:]):
        probability *= bigram_probability(
            previous_word, word, unigram_counts, bigram_counts
        )

    return probability


def run_bigram_language_model():
    corpus = [
        "<s> I love NLP </s>",
        "<s> I love deep learning </s>",
        "<s> deep learning is fun </s>",
    ]

    unigram_counts, bigram_counts = train_bigram_model(corpus)

    print("\nPart II - Q1: Bigram Language Model")
    print("\nUnigram counts:")
    for word, count in unigram_counts.items():
        print(f"{word}: {count}")

    print("\nBigram counts and MLE probabilities:")
    for (previous_word, word), count in bigram_counts.items():
        prob = bigram_probability(
            previous_word, word, unigram_counts, bigram_counts
        )
        print(f"({previous_word}, {word}): count={count}, probability={prob:.4f}")

    s1 = "<s> I love NLP </s>"
    s2 = "<s> I love deep learning </s>"

    p1 = sentence_probability(s1, unigram_counts, bigram_counts)
    p2 = sentence_probability(s2, unigram_counts, bigram_counts)

    print(f"\nP({s1}) = {p1:.6f}")
    print(f"P({s2}) = {p2:.6f}")

    if p1 > p2:
        print("Preferred sentence: <s> I love NLP </s>")
        print("Reason: It has the higher bigram probability.")
    elif p2 > p1:
        print("Preferred sentence: <s> I love deep learning </s>")
        print("Reason: It has the higher bigram probability.")
    else:
        print("Both sentences have the same probability.")


if __name__ == "__main__":
    matrix = [
        [5, 10, 5],
        [15, 20, 10],
        [0, 15, 10],
    ]
    confusion_matrix_metrics(matrix, ["Cat", "Dog", "Rabbit"])
    run_bigram_language_model()
