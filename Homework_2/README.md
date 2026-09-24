# CS5760 Natural Language Processing - Homework 2

**Student Name:** Shaik Ameer Pasha  
**Student ID:** 700782176  
**Course:** CS5760 Natural Language Processing  
**Semester:** Fall 2026

## Part I - Writing Calculation

### Q1. Worked Example Document Classification

Test document: `predictable no fun`

Using the worked Naive Bayes example from the course slides:

- `P(-) = 3/5`, `P(+) = 2/5`
- Vocabulary size `|V| = 20`
- Negative token total `n- = 14`
- Positive token total `n+ = 9`

Smoothed likelihoods for the negative class:

- `P(predictable | -) = (1+1)/(14+20) = 2/34`
- `P(no | -) = (1+1)/(14+20) = 2/34`
- `P(fun | -) = (0+1)/(14+20) = 1/34`

Negative score:

`P(-) x P(predictable|-) x P(no|-) x P(fun|-)`

`= (3/5) x (2/34) x (2/34) x (1/34)`

`≈ 6.1 x 10^-5`

Smoothed likelihoods for the positive class:

- `P(predictable | +) = (0+1)/(9+20) = 1/29`
- `P(no | +) = (0+1)/(9+20) = 1/29`
- `P(fun | +) = (1+1)/(9+20) = 2/29`

Positive score:

`P(+) x P(predictable|+) x P(no|+) x P(fun|+)`

`= (2/5) x (1/29) x (1/29) x (2/29)`

`≈ 3.2 x 10^-5`

**Answer:** Since `6.1 x 10^-5 > 3.2 x 10^-5`, the document is classified as **Negative (-)**.

### Q2. Harms of Classification

**1. Define representational harm and explain how Kiritchenko & Mohammad (2018) demonstrates it.**

Representational harm occurs when a classification system reinforces stereotypes or represents social groups unfairly. In the course example, Kiritchenko & Mohammad (2018) studied sentiment systems using sentence pairs that differed in names associated with different racial groups. The systems tended to assign lower sentiment or more negative emotion to sentences containing names associated with African Americans. This demonstrates how a classifier can perpetuate harmful social stereotypes even when the sentence meaning is otherwise similar.

**2. What is one risk of censorship in toxicity classification systems?**

A toxicity classifier can incorrectly flag non-toxic text simply because it contains minority identity terms such as terms referring to disability or sexual identity. As discussed with Dixon et al. (2018) and Oliva et al. (2021), this can suppress legitimate speech and reduce the visibility of content from or about minority communities.

**3. Why may classifiers perform worse on African American English or Indian English?**

One reason is insufficient representation of these English varieties in the training data. If a model is trained mainly on more commonly represented varieties of English, differences in vocabulary, spelling, syntax, and usage in African American English or Indian English may be learned poorly, producing lower classification accuracy.

### Q3. Bigram Probabilities and the Zero-Probability Problem

Training corpus:

- `<s> I love NLP </s>`
- `<s> I love deep learning </s>`
- `<s> deep learning is fun </s>`

**1. Probability of S1: `<s> I love NLP </s>`**

`P(S1) = P(I|<s>) x P(love|I) x P(NLP|love) x P(</s>|NLP)`

`= (2/3) x (2/2) x (1/2) x (1/1)`

`= 1/3 ≈ 0.3333`

**2. Probability of S2: `<s> I love deep learning </s>`**

`P(S2) = P(I|<s>) x P(love|I) x P(deep|love) x P(learning|deep) x P(</s>|learning)`

`= (2/3) x (2/2) x (1/2) x (2/2) x (1/2)`

`= 1/6 ≈ 0.1667`

**3. Which sentence is more probable?**

S1 is more probable because `0.3333 > 0.1667`.

**4. Compute `P(noodle|ate)` using MLE.**

`P(noodle|ate) = count(ate, noodle) / count(ate) = 0/12 = 0`

**5. Why is zero probability a problem?**

If one bigram has probability zero, multiplying the bigram probabilities makes the probability of the entire sentence zero. It also creates a problem for perplexity because the model has assigned zero probability to the observed sequence.

**6. Apply Add-1 smoothing with vocabulary size 10 and count after `ate` equal to 12.**

`P(noodle|ate) = (0+1)/(12+10) = 1/22 ≈ 0.0455`

### Q4. Backoff Model

Training corpus:

- `<s> I like cats </s>`
- `<s> I like dogs </s>`
- `<s> You like cats </s>`

**1. Compute `P(cats|I,like)`.**

`P(cats|I,like) = count(I like cats) / count(I like) = 1/2 = 0.5`

**2. Compute `P(dogs|You,like)` using trigram -> bigram backoff.**

The trigram `You like dogs` is unseen. Back off to the bigram probability:

`P(dogs|like) = count(like dogs) / count(like) = 1/3 ≈ 0.3333`

**3. Why is backoff necessary?**

The small training corpus does not contain every possible trigram. Backoff allows the model to use a lower-order n-gram when the higher-order n-gram is unseen, avoiding an unusable zero estimate in this example.

### Q5. Evaluation Metrics from a Multi-Class Confusion Matrix

Confusion matrix (rows = system output, columns = gold label):

| System / Gold | Cat | Dog | Rabbit |
|---|---:|---:|---:|
| Cat | 5 | 10 | 5 |
| Dog | 15 | 20 | 10 |
| Rabbit | 0 | 15 | 10 |

**Per-class metrics**

- Cat: Precision `= 5/20 = 0.2500`; Recall `= 5/20 = 0.2500`
- Dog: Precision `= 20/45 = 0.4444`; Recall `= 20/45 = 0.4444`
- Rabbit: Precision `= 10/25 = 0.4000`; Recall `= 10/25 = 0.4000`

**Macro averages**

- Macro Precision `= (0.2500 + 0.4444 + 0.4000)/3 ≈ 0.3648`
- Macro Recall `= (0.2500 + 0.4444 + 0.4000)/3 ≈ 0.3648`

**Micro averages**

- Total correct `= 5 + 20 + 10 = 35`
- Total examples `= 90`
- Micro Precision `= 35/90 ≈ 0.3889`
- Micro Recall `= 35/90 ≈ 0.3889`

**Interpretation:** Macro averaging gives equal importance to each class by averaging the class-level metrics. Micro averaging pools all predictions before computing the metric, so classes with more examples have greater influence.

**Programming implementation:** The required Python implementation that accepts the confusion matrix, calculates per-class precision/recall, calculates macro/micro precision/recall, and prints the results is in `700782176_homework2.py`.

## Part II - Programming

### Q1. Bigram Language Model Implementation

The required program is implemented in `700782176_homework2.py`.

It:

1. Reads the three supplied training sentences.
2. Computes unigram and bigram counts.
3. Estimates MLE bigram probabilities.
4. Implements a function to calculate the probability of a supplied sentence.
5. Tests both required sentences.
6. Prints which sentence is preferred and why.

Results:

- `P(<s> I love NLP </s>) = 0.333333`
- `P(<s> I love deep learning </s>) = 0.166667`

**Answer:** The model prefers `<s> I love NLP </s>` because it has the higher bigram probability.

## Program Files

- `700782176_homework2.py` - Python solutions for Part I Q5 and Part II Q1.
- `sample_output.txt` - Output produced by running the program.
- `Homework 2_Completed.docx` - Written answers to every assignment question.

## Run the Program

```bash
python 700782176_homework2.py
```


## Actual Program Output

The following is the actual output produced by running `700782176_homework2.py`:

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
