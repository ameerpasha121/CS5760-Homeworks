# CS5760 Natural Language Processing - Homework 2

**Student Name:** Shaik Ameer Pasha  
**Student ID:** 700782176  
**Course:** CS5760 Natural Language Processing  
**Semester:** Fall 2026

## Files

- `700782176_homework2.py` - Python implementation for Part I Q5 and Part II Q1.
- `sample_output.txt` - Actual output from running the Python program.
- `Homework 2_Answered.docx` - Completed written homework.

## Part I Q5 - Evaluation Metrics

The program accepts the supplied 3x3 confusion matrix and computes per-class precision and recall, macro-averaged precision and recall, and micro-averaged precision and recall.

Computed results:

- Cat: Precision = 0.2500, Recall = 0.2500
- Dog: Precision = 0.4444, Recall = 0.4444
- Rabbit: Precision = 0.4000, Recall = 0.4000
- Macro Precision = 0.3648
- Macro Recall = 0.3648
- Micro Precision = 0.3889
- Micro Recall = 0.3889

## Part II Q1 - Bigram Language Model

The program reads the three supplied training sentences, computes unigram and bigram counts, estimates MLE bigram probabilities, and provides a function for calculating the probability of a sentence.

Test results:

- `P(<s> I love NLP </s>) = 0.333333`
- `P(<s> I love deep learning </s>) = 0.166667`

The model prefers `<s> I love NLP </s>` because it has the higher probability.

## Actual Program Output

```text
Part I - Q5: Confusion Matrix Metrics
Cat: Precision = 0.2500, Recall = 0.2500
Dog: Precision = 0.4444, Recall = 0.4444
Rabbit: Precision = 0.4000, Recall = 0.4000
Macro Precision = 0.3648
Macro Recall = 0.3648
Micro Precision = 0.3889
Micro Recall = 0.3889

Part II - Q1: Bigram Language Model

Unigram counts:
<s>: 3
I: 2
love: 2
NLP: 1
</s>: 3
deep: 2
learning: 2
is: 1
fun: 1

Bigram counts and MLE probabilities:
(<s>, I): count=2, probability=0.6667
(I, love): count=2, probability=1.0000
(love, NLP): count=1, probability=0.5000
(NLP, </s>): count=1, probability=1.0000
(love, deep): count=1, probability=0.5000
(deep, learning): count=2, probability=1.0000
(learning, </s>): count=1, probability=0.5000
(<s>, deep): count=1, probability=0.3333
(learning, is): count=1, probability=0.5000
(is, fun): count=1, probability=1.0000
(fun, </s>): count=1, probability=1.0000

P(<s> I love NLP </s>) = 0.333333
P(<s> I love deep learning </s>) = 0.166667
Preferred sentence: <s> I love NLP </s>
Reason: It has the higher bigram probability.
```

## Part I Q1 - Worked Naive Bayes Classification

Using the numerical values from the course slide:

- `P(-) = 3/5`, `P(+) = 2/5`
- `|V| = 20`
- Negative token total = 14
- Positive token total = 9
- `P(predictable|-) = 2/34`
- `P(no|-) = 2/34`
- `P(fun|-) = 1/34`
- `P(predictable|+) = 1/29`
- `P(no|+) = 1/29`
- `P(fun|+) = 2/29`

Scores:

- Negative: `(3/5) × (2/34) × (2/34) × (1/34) ≈ 6.1 × 10^-5`
- Positive: `(2/5) × (1/29) × (1/29) × (2/29) ≈ 3.2 × 10^-5`

Therefore, `predictable no fun` is classified as **Negative (-)**.

## Run

```bash
python 700782176_homework2.py
```
