# Probabilistic-Spell-Checker-The-Noisy-Channel
### Spell checker that handles both non-word and real-word errors given in a context
The purpose of this assignment is to build probabilistic language models by implementing the full algorithmic pipeline.

Specifics:
- Use the noisy channel model to correct the errors: 
  - A correction of a misspelled word depends on both the most probable correction on the error type-character level and the word prior 
  - A correction of a word in a sentence depends on the error type-character level and on the language model -- the correction should     maximize the likelihood of getting the full corrected sentence. 
- Use the language model when correcting words in a context.

References:
- [Spelling Correction and the Noisy Channel](https://web.stanford.edu/~jurafsky/slp3/B.pdf)
- [Markov Language Model](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
