# kaggle-quora-insincere-questions
Kaggle Quora Insincere Questions Classification

## Ideas
- [ ] Train on the entire train set

To do ensemble of multiple different models, waiting for k-fold training is not reasonable.

- [ ] Optimize for loss/f1

F1 is dependent on a threshold. Rather, optimize the model for loss is very straight forward. Test both.

- [ ] Balanced sampling

The data set is very skewed. Balanced sampling would help. You can gradually increase the ratio of
positive samples. Try both static/dynamic sampling ratio scheduling.

- [ ] Cyclic learning rate

For faster convergence

- [ ] Ensemble of embeddings

Glove, Paragram, fastText embeddings are ready. Try both (1) average of embeddings and (2) ensemble of
models with different embeddings.

- [ ] Transformer

Of course, do it. Refer to - https://qiita.com/halhorn/items/c91497522be27bde17ce

- [ ] Other NLP features - number of words

Might be effective.

- [ ] Pre-processing of text

Try this kernel - https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing