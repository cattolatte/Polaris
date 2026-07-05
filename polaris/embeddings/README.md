# Embeddings Module

Load **pretrained word vectors** and align them to a Polaris `Vocabulary`, so a
model's embedding layer can start from GloVe instead of random initialization
(v0.10). Benchmarks showed this does not break the IMDB ceiling — pretrained *word*
vectors help most when labeled data is scarce — but the capability is a clean,
reusable piece.

## Public surface

- `load_glove(path)` (`glove.py`) — read a GloVe `.txt` file into a mapping from
  token to vector tensor. Heavy/optional (`torch`) imports are lazy.
- `build_embedding_matrix(vocabulary, vectors, *, embedding_dim, seed=0)`
  (`glove.py`) — build a `(vocab_size, embedding_dim)` matrix aligned to the
  vocabulary: the GloVe vector for each known word (with a lowercase fallback),
  seeded-random for out-of-vocabulary tokens, and zeros for padding.

Pass the result as `pretrained_embeddings` to `MeanPoolingClassifier` or
`TransformerEncoderClassifier`.
