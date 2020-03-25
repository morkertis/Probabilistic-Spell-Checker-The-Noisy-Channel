# The Noisy Channel and a Probabilistic Spell Checker
### Spell checker that handles both non-word and real-word errors given in a context
The purpose of this assignment is to give a hands on experience with probabilistic models of language by implementing the full algorithmic pipeline.

- Use the noisy channel model to correct the errors: 
- - A correction of a misspelled word depends on both the most probable correction on the error type-character level and the word prior 
- - A correction of a word in a sentence depends on the error type-character level and on the language model -- the correction should     maximize the likelihood of getting the full corrected sentence. 
- Use the language model when correcting words in a sentential context.

References:
- [A Spelling Correction Program Based on a Noisy Channel Model](https://dl.acm.org/doi/pdf/10.3115/997939.997975)
