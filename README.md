# CS5760 Natural Language Processing - Homework 1

**Student Name:** Shaik Ameer Pasha  
**Student ID:** 700782176  
**Semester:** Fall 2026

This repository contains the written answers and Python implementation for Homework 1. The main source file is `700782176_homework1.py`. It includes regex examples, a mini Byte Pair Encoding (BPE) learner, BPE segmentation, and tokenization comparisons.

---

## Q1. Regex

### 1. U.S. ZIP codes

Regex:

```regex
(?<!\w)\d{5}(?:[- ]\d{4})?(?!\w)
```

This matches `12345`, `12345-6789`, and `12345 6789`. The lookarounds prevent a ZIP code from being matched inside a longer word or number.

### 2. Words that do not start with a capital letter

Regex:

```regex
\b[a-z][A-Za-z]*(?:['’\-][A-Za-z]+)*\b
```

The first character must be lowercase. Internal apostrophes and hyphens are allowed, so words such as `don't` and `state-of-the-art` are matched.

### 3. Numbers

Regex:

```regex
(?<!\w)[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?!\w)
```

Examples matched include `-12`, `+1,234.56`, `42`, and `1.23e-4`.

### 4. Variants of email

Regex:

```regex
\be(?:mail|[-– ]mail)\b
```

Use the case-insensitive flag. It matches `email`, `e-mail`, `e mail`, `e–mail`, `EMAIL`, and related case variants.

### 5. go, goo, gooo, ... with optional punctuation

Regex:

```regex
\bgo+(?:[!.,?])?(?!\w)
```

It requires at least one `o` and permits one optional final punctuation mark.

### 6. Lines ending with a question mark and optional closing quotes/brackets

Regex:

```regex
\?[\"'”’\)\]]*\s*$
```

Use multiline mode. The line must end with `?`, possibly followed only by closing quotation/bracket characters and spaces.

---

## Q2. Manual BPE on a Toy Corpus

### Q2.1 Initial representation and first three merges

Toy corpus frequencies:

- `low` x 5
- `lowest` x 2
- `newer` x 6
- `wider` x 3
- `new` x 2

With the end-of-word marker `_`, the initial word forms are:

```text
low     -> l o w _
lowest  -> l o w e s t _
newer   -> n e w e r _
wider   -> w i d e r _
new     -> n e w _
```

Initial vocabulary:

```text
{ _, d, e, i, l, n, o, r, s, t, w }
```

There are 11 initial tokens.

#### Step 1

The pair `(e, r)` occurs 9 times. The pair `(r, _)` is also tied at 9; I use `(e, r)` as the first merge.

```text
e + r -> er
```

Updated corpus examples:

```text
newer x 6 -> n e w er _
wider x 3 -> w i d er _
lowest x 2 -> l o w e s t _
```

New token: `er`  
Updated vocabulary size: 12

#### Step 2

The most frequent pair is `(er, _)`, count 9.

```text
er + _ -> er_
```

Updated corpus examples:

```text
newer x 6 -> n e w er_
wider x 3 -> w i d er_
new x 2   -> n e w _
```

New token: `er_`  
Updated vocabulary size: 13

#### Step 3

The pairs `(n, e)` and `(e, w)` are tied at count 8. I choose `(n, e)`.

```text
n + e -> ne
```

Updated corpus examples:

```text
newer x 6 -> ne w er_
new x 2   -> ne w _
wider x 3 -> w i d er_
```

New token: `ne`  
Updated vocabulary size: 14

### Q2.2 Mini-BPE learner

The Python program learns 15 merges for the toy corpus. The sequence produced by the program is:

| Step | Pair | Count | New token | Vocabulary size |
|---:|---|---:|---|---:|
| 1 | `(e, r)` | 9 | `er` | 12 |
| 2 | `(er, _)` | 9 | `er_` | 13 |
| 3 | `(n, e)` | 8 | `ne` | 14 |
| 4 | `(ne, w)` | 8 | `new` | 15 |
| 5 | `(l, o)` | 7 | `lo` | 16 |
| 6 | `(lo, w)` | 7 | `low` | 17 |
| 7 | `(new, er_)` | 6 | `newer_` | 18 |
| 8 | `(low, _)` | 5 | `low_` | 19 |
| 9 | `(w, i)` | 3 | `wi` | 20 |
| 10 | `(wi, d)` | 3 | `wid` | 21 |
| 11 | `(wid, er_)` | 3 | `wider_` | 22 |
| 12 | `(low, e)` | 2 | `lowe` | 23 |
| 13 | `(lowe, s)` | 2 | `lowes` | 24 |
| 14 | `(lowes, t)` | 2 | `lowest` | 25 |
| 15 | `(lowest, _)` | 2 | `lowest_` | 26 |

Requested segmentation after these merges:

```text
new        -> new _
newer      -> newer_
lowest     -> lowest_
widest     -> wid e s t _
newestest  -> new e s t e s t _
```

Subword tokenization reduces the OOV problem because a word does not need to appear as a complete item in the training vocabulary. If a new word is unseen, BPE can represent it with smaller learned pieces and, when necessary, individual characters. For example, `widest` was not in the toy training corpus, but it can still be represented as `wid e s t _`. The invented word `newestest` can also be represented using the learned token `new` plus smaller pieces. A meaningful morpheme appears in `er_`, which corresponds to the English comparative suffix in words such as `newer` and `wider`. Therefore, BPE provides useful coverage while still learning larger recurring patterns.

### Q2.3 BPE on an English paragraph

Paragraph used:

> Students study language data in the university laboratory. The students analyze text data and build language models. A researcher carefully evaluates the models and compares tokenization methods. Modern tokenization helps computers process unfamiliar words and meaningful word forms. Researchers improve reliable language systems through repeated experiments.

The program learns 30 merges using `_` as the end-of-word marker.

Five most frequent merges at the start of training:

| Rank | Pair | Count | New token |
|---:|---|---:|---|
| 1 | `(s, _)` | 15 | `s_` |
| 2 | `(e, _)` | 9 | `e_` |
| 3 | `(a, n)` | 8 | `an` |
| 4 | `(a, t)` | 7 | `at` |
| 5 | `(e, r)` | 6 | `er` |

Five longest active subword tokens after 30 merges:

```text
language_
ents_
stud
the_
and_
```

Five segmented words:

```text
students     -> stud ents_
tokenization -> t o k en i z at i o n_
researcher   -> re s e ar c h er _
researchers  -> re s e ar c h er s_
experiments  -> e x p er i m ents_
```

The learned subwords include whole frequent words, stems, suffix-like pieces, and smaller character sequences. For example, `language_` becomes a complete word token because it occurs several times, while `stud` behaves like a stem and `ents_` behaves like a recurring word-ending unit. One advantage of subword tokenization is that rare words can still be represented without an unknown-word token. Another advantage is that common stems or endings can be reused across related words. A disadvantage is that a word can be broken into pieces that do not correspond to true linguistic morphemes. Another disadvantage is that rare or long words may become long token sequences, increasing sequence length. For English, BPE is useful because it balances vocabulary size with the ability to represent new forms.

---

## Q3. Bayes Rule Applied to Text

For document classification, Bayes' rule can be written as:

```text
P(c | d) = P(d | c) P(c) / P(d)
```

The class selected by MAP classification is the class with the largest value of:

```text
P(c) P(d | c)
```

**P(c)** is the prior probability of class `c`. It describes how likely the class is before looking at the current document. For example, if 60% of training documents are negative, the negative prior is 0.60.

**P(d | c)** is the likelihood of observing document `d` assuming the document belongs to class `c`. In a text classifier, it measures how well the words in the document fit that class.

**P(c | d)** is the posterior probability of class `c` after observing document `d`. It is the quantity we want when deciding which class is most likely for the document.

The denominator **P(d)** can be ignored when comparing classes because the document `d` is fixed. Therefore, `P(d)` has the same value for every candidate class. Dividing every class score by the same positive number does not change which class has the largest score.

---

## Q4. Add-1 Smoothing

Given:

```text
P(-) = 3/5
P(+) = 2/5
Vocabulary size V = 20
Negative-class token count = 14
```

### 1. Denominator for the negative class

For add-1 smoothing:

```text
denominator = total negative tokens + vocabulary size
            = 14 + 20
            = 34
```

**Answer: 34**

### 2. P(predictable | -)

`predictable` appears 2 times in the negative documents.

```text
P(predictable | -) = (2 + 1) / 34
                   = 3 / 34
                   = 0.0882 approximately
```

**Answer: 3/34, approximately 0.0882**

### 3. P(fun | -)

`fun` appears 0 times in the negative documents.

```text
P(fun | -) = (0 + 1) / 34
           = 1 / 34
           = 0.0294 approximately
```

**Answer: 1/34, approximately 0.0294**

Add-1 smoothing gives an unseen word a small nonzero probability instead of probability zero.

---

## Q5. Programming Question - Tokenization

### 1. Paragraph and naive tokenization

Paragraph:

> UCM students don't ignore data quality. A researcher's model can improve quickly, but it's not perfect. New York researchers use state-of-the-art tools for language analysis. In the long run, well-tested systems make better decisions.

Naive space-based tokens:

```text
['UCM', 'students', "don't", 'ignore', 'data', 'quality.', 'A', "researcher's",
 'model', 'can', 'improve', 'quickly,', 'but', "it's", 'not', 'perfect.', 'New',
 'York', 'researchers', 'use', 'state-of-the-art', 'tools', 'for', 'language',
 'analysis.', 'In', 'the', 'long', 'run,', 'well-tested', 'systems', 'make',
 'better', 'decisions.']
```

Manual corrected tokens:

```text
['UCM', 'student', '-s', 'do', "n't", 'ignore', 'data', 'quality', '.', 'A',
 'researcher', "'s", 'model', 'can', 'improve', 'quickly', ',', 'but', 'it', "'s",
 'not', 'perfect', '.', 'New', 'York', 'researcher', '-s', 'use',
 'state-of-the-art', 'tool', '-s', 'for', 'language', 'analysis', '.', 'In', 'the',
 'long', 'run', ',', 'well-tested', 'system', '-s', 'make', 'better', 'decision',
 '-s', '.']
```

Main corrections:

- Sentence punctuation is separated: `quality.` becomes `quality` and `.`.
- Commas are separated: `quickly,` becomes `quickly` and `,`.
- The clitic in `don't` is separated as `do` + `n't`.
- Possessive/contraction clitics are separated: `researcher's` -> `researcher` + `'s`, and `it's` -> `it` + `'s`.
- Selected plural suffixes are shown separately, such as `students` -> `student` + `-s`.
- Hyphenated compounds such as `state-of-the-art` and `well-tested` are kept together in the manual version because they function as combined lexical units in this paragraph.

### 2. Comparison with an NLP tool

I used NLTK's `TreebankWordTokenizer` because it can run directly without downloading a separate language model.

Tool output:

```text
['UCM', 'students', 'do', "n't", 'ignore', 'data', 'quality.', 'A', 'researcher',
 "'s", 'model', 'can', 'improve', 'quickly', ',', 'but', 'it', "'s", 'not',
 'perfect.', 'New', 'York', 'researchers', 'use', 'state-of-the-art', 'tools',
 'for', 'language', 'analysis.', 'In', 'the', 'long', 'run', ',', 'well-tested',
 'systems', 'make', 'better', 'decisions', '.']
```

The NLTK tokenizer agrees with the manual analysis on important clitics: it splits `don't`, `researcher's`, and `it's`. It also separates the commas in `quickly,` and `run,`. However, it does not separate plural suffixes such as the `-s` in `students` or `researchers`, because standard word tokenization normally treats inflected words as one token rather than performing morphological segmentation. It also keeps the hyphenated compounds as one token. In this particular multi-sentence input, TreebankWordTokenizer leaves some internal sentence periods attached while separating the final period; this is a behavior of this tokenizer and illustrates that tool tokenization rules can differ from a manually designed scheme.

### 3. Multiword Expressions (MWEs)

1. **New York** - This is a place name. Treating it as one expression preserves the identity of the location instead of interpreting `New` and `York` independently.
2. **state-of-the-art** - This fixed expression means highly advanced or modern. Its full meaning is not obtained by processing each component independently.
3. **in the long run** - This idiomatic phrase means over a long period of time. Treating the phrase as one semantic unit preserves the intended meaning.

### 4. Reflection

The hardest part of English tokenization in this example was deciding what should remain a single token and what should be split. Simple space tokenization fails because punctuation stays attached to words and contractions contain more than one grammatical unit. English is easier than many morphologically rich languages because word boundaries are often marked by spaces, but apostrophes, hyphens, and inflection still create ambiguity. Punctuation makes tokenization harder because the same symbol can have different roles depending on context. Morphology creates another choice because a system may keep `students` as one token or split it into a stem and suffix. Multiword expressions are also difficult because several space-separated words may represent one meaning. A good tokenizer therefore needs rules that match the goals of the NLP application rather than relying only on spaces.

---

## How to Run

Install NLTK if it is not already installed:

```bash
pip install nltk
```

Run the program:

```bash
python 700782176_homework1.py
```

The program prints regex checks, BPE merge information, BPE segmentation, and the three tokenization outputs used in this README.

---

## Actual Program Output

The following is the actual output produced by running:

```bash
python 700782176_homework1.py
```

This output documents the programming results required for **Q1, Q2.2, Q2.3, and Q5**.

```text
Q1 - Regex checks
ZIP: ['12345', '12345-6789', '12345 6789']
Non-capital words: ['apple', "don't", 'state-of-the-art']
Numbers: ['-12', '+1,234.56', '1.23e-4', '42']
Email: ['email', 'E-MAIL', 'e mail', 'e–mail']
Go: ['go', 'goo', 'gooo!', 'goooo?']
Question lines: ['?', '?”']

Q2.2 - Toy corpus BPE
Step  1: top pair=('e', 'r'), count=9, new token='er', vocabulary size=12
Step  2: top pair=('er', '_'), count=9, new token='er_', vocabulary size=13
Step  3: top pair=('n', 'e'), count=8, new token='ne', vocabulary size=14
Step  4: top pair=('ne', 'w'), count=8, new token='new', vocabulary size=15
Step  5: top pair=('l', 'o'), count=7, new token='lo', vocabulary size=16
Step  6: top pair=('lo', 'w'), count=7, new token='low', vocabulary size=17
Step  7: top pair=('new', 'er_'), count=6, new token='newer_', vocabulary size=18
Step  8: top pair=('low', '_'), count=5, new token='low_', vocabulary size=19
Step  9: top pair=('w', 'i'), count=3, new token='wi', vocabulary size=20
Step 10: top pair=('wi', 'd'), count=3, new token='wid', vocabulary size=21
Step 11: top pair=('wid', 'er_'), count=3, new token='wider_', vocabulary size=22
Step 12: top pair=('low', 'e'), count=2, new token='lowe', vocabulary size=23
Step 13: top pair=('lowe', 's'), count=2, new token='lowes', vocabulary size=24
Step 14: top pair=('lowes', 't'), count=2, new token='lowest', vocabulary size=25
Step 15: top pair=('lowest', '_'), count=2, new token='lowest_', vocabulary size=26
new        -> new _
newer      -> newer_
lowest     -> lowest_
widest     -> wid e s t _
newestest  -> new e s t e s t _

Q2.3 - English paragraph BPE
Step  1: top pair=('s', '_'), count=15, new token='s_', vocabulary size=26
Step  2: top pair=('e', '_'), count=9, new token='e_', vocabulary size=27
Step  3: top pair=('a', 'n'), count=8, new token='an', vocabulary size=28
Step  4: top pair=('a', 't'), count=7, new token='at', vocabulary size=29
Step  5: top pair=('e', 'r'), count=6, new token='er', vocabulary size=30
Step  6: top pair=('d', '_'), count=6, new token='d_', vocabulary size=31
Step  7: top pair=('r', 'e'), count=6, new token='re', vocabulary size=32
Step  8: top pair=('e', 'n'), count=5, new token='en', vocabulary size=33
Step  9: top pair=('t', 'h'), count=5, new token='th', vocabulary size=34
Step 10: top pair=('o', 'r'), count=5, new token='or', vocabulary size=35
Step 11: top pair=('s', 't'), count=4, new token='st', vocabulary size=36
Step 12: top pair=('y', '_'), count=4, new token='y_', vocabulary size=37
Step 13: top pair=('n', '_'), count=4, new token='n_', vocabulary size=38
Step 14: top pair=('o', 'd'), count=4, new token='od', vocabulary size=39
Step 15: top pair=('st', 'u'), count=3, new token='stu', vocabulary size=40
Step 16: top pair=('stu', 'd'), count=3, new token='stud', vocabulary size=41
Step 17: top pair=('en', 't'), count=3, new token='ent', vocabulary size=42
Step 18: top pair=('ent', 's_'), count=3, new token='ents_', vocabulary size=43
Step 19: top pair=('l', 'an'), count=3, new token='lan', vocabulary size=44
Step 20: top pair=('lan', 'g'), count=3, new token='lang', vocabulary size=45
Step 21: top pair=('lang', 'u'), count=3, new token='langu', vocabulary size=46
Step 22: top pair=('langu', 'a'), count=3, new token='langua', vocabulary size=47
Step 23: top pair=('langua', 'g'), count=3, new token='languag', vocabulary size=48
Step 24: top pair=('languag', 'e_'), count=3, new token='language_', vocabulary size=49
Step 25: top pair=('a', '_'), count=3, new token='a_', vocabulary size=50
Step 26: top pair=('th', 'e_'), count=3, new token='the_', vocabulary size=51
Step 27: top pair=('an', 'd_'), count=3, new token='and_', vocabulary size=52
Step 28: top pair=('m', 'od'), count=3, new token='mod', vocabulary size=53
Step 29: top pair=('e', 'l'), count=3, new token='el', vocabulary size=54
Step 30: top pair=('a', 'r'), count=3, new token='ar', vocabulary size=55

Five most frequent merges:
1. ('s', '_') -> s_ (count=15)
2. ('e', '_') -> e_ (count=9)
3. ('a', 'n') -> an (count=8)
4. ('a', 't') -> at (count=7)
5. ('e', 'r') -> er (count=6)
Five longest resulting active subword tokens: ['language_', 'ents_', 'stud', 'the_', 'and_']

Five segmented words:
students     -> stud ents_
tokenization -> t o k en i z at i o n_
researcher   -> re s e ar c h er _
researchers  -> re s e ar c h er s_
experiments  -> e x p er i m ents_

Q5 - Tokenization
Paragraph: UCM students don't ignore data quality. A researcher's model can improve quickly, but it's not perfect. New York researchers use state-of-the-art tools for language analysis. In the long run, well-tested systems make better decisions.

Naive space-based tokens:
['UCM', 'students', "don't", 'ignore', 'data', 'quality.', 'A', "researcher's", 'model', 'can', 'improve', 'quickly,', 'but', "it's", 'not', 'perfect.', 'New', 'York', 'researchers', 'use', 'state-of-the-art', 'tools', 'for', 'language', 'analysis.', 'In', 'the', 'long', 'run,', 'well-tested', 'systems', 'make', 'better', 'decisions.']

Manual tokens:
['UCM', 'student', '-s', 'do', "n't", 'ignore', 'data', 'quality', '.', 'A', 'researcher', "'s", 'model', 'can', 'improve', 'quickly', ',', 'but', 'it', "'s", 'not', 'perfect', '.', 'New', 'York', 'researcher', '-s', 'use', 'state-of-the-art', 'tool', '-s', 'for', 'language', 'analysis', '.', 'In', 'the', 'long', 'run', ',', 'well-tested', 'system', '-s', 'make', 'better', 'decision', '-s', '.']

NLTK TreebankWordTokenizer tokens:
['UCM', 'students', 'do', "n't", 'ignore', 'data', 'quality.', 'A', 'researcher', "'s", 'model', 'can', 'improve', 'quickly', ',', 'but', 'it', "'s", 'not', 'perfect.', 'New', 'York', 'researchers', 'use', 'state-of-the-art', 'tools', 'for', 'language', 'analysis.', 'In', 'the', 'long', 'run', ',', 'well-tested', 'systems', 'make', 'better', 'decisions', '.']
```
