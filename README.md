Implementation in Rust of BetweenCut (arXiv:2610.10075), heavy/light classification of every node in a public rooted tree.

```
cargo test
```

Not implemented: the TCT privacy proof, Appendix D, Appendix E, and Algorithm 5. There is no privacy audit. Nodes at one depth are visited in increasing index order. Laplace noise is the inverse CDF of a local xorshift64, not a named RNG.
