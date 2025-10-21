# Tokenizer Benchmark Report

This report details the performance of different `encode` methods in the `HuggingFaceTokenizer` and `RustBPETokenizer` classes. Each method was benchmarked on three different types of input: a short string, a long string, and a list of strings. The times shown are the total time taken for 1000 iterations.

## HuggingFaceTokenizer

| Input Type      | Time (seconds) |
|-----------------|----------------|
| Short Text | 0.028929       |
| Long Text | 0.074444       |
| List Of Texts | 0.098034       |

## RustBPETokenizer

| Method            | Short Text     | Long Text      | List of Texts  |
|-------------------|----------------|----------------|----------------|
| `encode` | 0.018362       | 0.021349       | 1.945049       |
| `encode_optimized` | 0.006990       | 0.021443       | 1.033614       |
| `encode_optimized2` | 0.007315       | 0.021685       | 1.051735       |
| `encode_optimized3` | 0.129471       | 0.030476       | 1.419208       |
| `encode_optimized4` | 0.008096       | 0.024128       | 1.411694       |

## Conclusion

Based on the results, the most efficient method for the `RustBPETokenizer` is **`encode_optimized`**. This method consistently performed the best across all input types.
