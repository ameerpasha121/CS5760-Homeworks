"""
CS5760 Natural Language Processing - Homework 1
Student Name: Shaik Ameer Pasha
Student ID: 700782176
Fall 2026

This file contains the programming portions and reproducible examples for
Regex, BPE, and tokenization tasks in Homework 1.
"""

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Q1. Regular expressions
# ---------------------------------------------------------------------------
ZIP_RE = re.compile(r"(?<!\w)\d{5}(?:[- ]\d{4})?(?!\w)")
NON_CAPITAL_WORD_RE = re.compile(r"\b[a-z][A-Za-z]*(?:['’\-][A-Za-z]+)*\b")
NUMBER_RE = re.compile(
    r"(?<!\w)[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?!\w)"
)
EMAIL_RE = re.compile(r"\be(?:mail|[-– ]mail)\b", re.IGNORECASE)
GO_RE = re.compile(r"\bgo+(?:[!.,?])?(?!\w)", re.IGNORECASE)
QUESTION_LINE_RE = re.compile(r"\?[\"'”’\)\]]*\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Q2. Mini BPE implementation
# ---------------------------------------------------------------------------
def build_word_frequency(text):
    """Return lowercase word frequencies using alphabetic/hyphen/apostrophe words."""
    words = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", text.lower())
    return Counter(words)


def initialize_bpe(word_freq):
    """Represent each word as characters followed by the end-of-word marker _."""
    return {tuple(list(word) + ["_"]): count for word, count in word_freq.items()}


def get_pair_counts(state):
    """Count adjacent token pairs, weighted by word frequency."""
    counts = Counter()
    for tokens, frequency in state.items():
        for pair in zip(tokens, tokens[1:]):
            counts[pair] += frequency
    return counts


def merge_pair(state, pair):
    """Merge every occurrence of one adjacent token pair."""
    merged_token = "".join(pair)
    new_state = {}

    for tokens, frequency in state.items():
        output = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                output.append(merged_token)
                i += 2
            else:
                output.append(tokens[i])
                i += 1
        new_state[tuple(output)] = frequency

    return new_state, merged_token


def learn_bpe(text, num_merges=10, verbose=True):
    """Learn BPE merges. Ties use the first pair encountered in corpus order."""
    word_freq = build_word_frequency(text)
    state = initialize_bpe(word_freq)
    vocabulary = {token for word in state for token in word}
    merges = []

    for step in range(1, num_merges + 1):
        pair_counts = get_pair_counts(state)
        if not pair_counts:
            break

        top_pair, count = pair_counts.most_common(1)[0]
        state, new_token = merge_pair(state, top_pair)
        vocabulary.add(new_token)
        merges.append(top_pair)

        if verbose:
            print(
                f"Step {step:2d}: top pair={top_pair}, count={count}, "
                f"new token='{new_token}', vocabulary size={len(vocabulary)}"
            )

    return merges, vocabulary, state


def segment_word(word, merges):
    """Apply learned merges to a new word and keep _ as the word-end marker."""
    tokens = list(word.lower()) + ["_"]

    for pair in merges:
        merged_token = "".join(pair)
        output = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                output.append(merged_token)
                i += 2
            else:
                output.append(tokens[i])
                i += 1
        tokens = output

    return tokens


TOY_CORPUS = (
    "low low low low low lowest lowest "
    "newer newer newer newer newer newer "
    "wider wider wider new new"
)

ENGLISH_PARAGRAPH = (
    "Students study language data in the university laboratory. "
    "The students analyze text data and build language models. "
    "A researcher carefully evaluates the models and compares tokenization methods. "
    "Modern tokenization helps computers process unfamiliar words and meaningful word forms. "
    "Researchers improve reliable language systems through repeated experiments."
)


# ---------------------------------------------------------------------------
# Q5. Tokenization comparison
# ---------------------------------------------------------------------------
TOKENIZATION_PARAGRAPH = (
    "UCM students don't ignore data quality. "
    "A researcher's model can improve quickly, but it's not perfect. "
    "New York researchers use state-of-the-art tools for language analysis. "
    "In the long run, well-tested systems make better decisions."
)


def naive_space_tokenize(text):
    """Split only on whitespace, so punctuation remains attached."""
    return text.split()


def manual_tokenize(text):
    """
    A small manual English tokenizer for this homework paragraph.
    It separates punctuation and clitics and shows plural -s as a suffix token.
    Hyphenated compounds are kept together because they act as one lexical unit here.
    """
    # First find words (including apostrophes/hyphens) or individual punctuation marks.
    raw = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[.,!?]", text)
    output = []

    for token in raw:
        lower = token.lower()

        # Split common English clitics used in the paragraph.
        if lower.endswith("n't") and len(token) > 3:
            # don't -> do + n't
            stem = token[:-3]
            if stem.lower() == "do":
                output.extend([stem, "n't"])
            else:
                output.extend([stem, "n't"])
        elif lower.endswith("'s") and len(token) > 2:
            output.extend([token[:-2], "'s"])
        # Split selected plural forms to illustrate inflectional suffix handling.
        elif lower in {"students", "researchers", "tools", "systems", "decisions"}:
            output.extend([token[:-1], "-s"])
        else:
            output.append(token)

    return output


def tool_tokenize(text):
    """Tokenize with NLTK's TreebankWordTokenizer (no corpus download required)."""
    try:
        from nltk.tokenize import TreebankWordTokenizer
    except ImportError as exc:
        raise RuntimeError("Install NLTK with: pip install nltk") from exc

    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)


def print_q1_examples():
    """Print small checks for the six regular expressions."""
    examples = {
        "ZIP": "Valid: 12345, 12345-6789, 12345 6789; invalid inside A12345B.",
        "Non-capital words": "apple Banana don't state-of-the-art Zebra",
        "Numbers": "-12 +1,234.56 1.23e-4 42",
        "Email": "email E-MAIL e mail e–mail",
        "Go": "go goo gooo! goooo?",
        "Question lines": 'Is this ready?\nShe asked, "Really?”\nThis is a statement.'
    }
    regexes = [ZIP_RE, NON_CAPITAL_WORD_RE, NUMBER_RE, EMAIL_RE, GO_RE, QUESTION_LINE_RE]

    for (label, text), regex in zip(examples.items(), regexes):
        print(f"{label}: {regex.findall(text)}")


def run_q2_2():
    """Learn 15 merges on the classroom toy corpus and segment requested words."""
    print("\nQ2.2 - Toy corpus BPE")
    merges, _, _ = learn_bpe(TOY_CORPUS, num_merges=15, verbose=True)
    for word in ["new", "newer", "lowest", "widest", "newestest"]:
        print(f"{word:10s} -> {' '.join(segment_word(word, merges))}")


def run_q2_3():
    """Learn 30 merges on the selected English paragraph and print requested results."""
    print("\nQ2.3 - English paragraph BPE")
    merges, vocabulary, state = learn_bpe(ENGLISH_PARAGRAPH, num_merges=30, verbose=True)

    # Recompute the first five merge counts for display.
    temp_state = initialize_bpe(build_word_frequency(ENGLISH_PARAGRAPH))
    print("\nFive most frequent merges:")
    for step in range(1, 6):
        pair_counts = get_pair_counts(temp_state)
        pair, count = pair_counts.most_common(1)[0]
        temp_state, token = merge_pair(temp_state, pair)
        print(f"{step}. {pair} -> {token} (count={count})")

    active_tokens = {token for word in state for token in word}
    longest = sorted(
        active_tokens,
        key=lambda token: (len(token.replace("_", "")), len(token), token),
        reverse=True,
    )[:5]
    print("Five longest resulting active subword tokens:", longest)

    print("\nFive segmented words:")
    for word in ["students", "tokenization", "researcher", "researchers", "experiments"]:
        print(f"{word:12s} -> {' '.join(segment_word(word, merges))}")


def run_q5():
    """Print naive, manual, and NLTK tokenization for the selected paragraph."""
    print("\nQ5 - Tokenization")
    print("Paragraph:", TOKENIZATION_PARAGRAPH)
    print("\nNaive space-based tokens:")
    print(naive_space_tokenize(TOKENIZATION_PARAGRAPH))
    print("\nManual tokens:")
    print(manual_tokenize(TOKENIZATION_PARAGRAPH))
    print("\nNLTK TreebankWordTokenizer tokens:")
    print(tool_tokenize(TOKENIZATION_PARAGRAPH))


if __name__ == "__main__":
    print("Q1 - Regex checks")
    print_q1_examples()
    run_q2_2()
    run_q2_3()
    run_q5()
