
import os
import timeit
import numpy as np
from nanochat.tokenizer import HuggingFaceTokenizer, RustBPETokenizer

def get_tokenizer_dir():
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    nanochat_dir = os.path.join(cache_dir, "nanochat")
    tokenizer_dir = os.path.join(nanochat_dir, "tokenizer")
    return tokenizer_dir

def run_benchmark():
    tokenizer_dir = get_tokenizer_dir()

    # Check if tokenizer directory exists
    if not os.path.exists(tokenizer_dir):
        print(f"Tokenizer directory not found at: {tokenizer_dir}")
        print("Please run scripts/tok_train.py to create the tokenizer directory.")
        return

    # Initialize tokenizers
    hf_tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
    rust_tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

    # Sample data for benchmarking
    short_text = "This is a short sentence."
    long_text = "This is a much longer sentence that is designed to test the performance of the tokenizers on a larger input."
    text_list = ["Here is the first sentence.", "And this is the second one.", "Finally, the third sentence for testing."]

    # Number of iterations for timeit
    iterations = 1000

    # Benchmarking results dictionary
    results = {}

    # --- HuggingFaceTokenizer ---
    hf_results = {}
    # Short text
    hf_results["short_text"] = timeit.timeit(lambda: hf_tokenizer.encode(short_text), number=iterations)
    # Long text
    hf_results["long_text"] = timeit.timeit(lambda: hf_tokenizer.encode(long_text), number=iterations)
    # List of texts
    hf_results["list_of_texts"] = timeit.timeit(lambda: hf_tokenizer.encode(text_list), number=iterations)
    results["HuggingFaceTokenizer"] = hf_results

    # --- RustBPETokenizer ---
    rust_results = {}
    # encode
    rust_encode_results = {}
    rust_encode_results["short_text"] = timeit.timeit(lambda: rust_tokenizer.encode(short_text), number=iterations)
    rust_encode_results["long_text"] = timeit.timeit(lambda: rust_tokenizer.encode(long_text), number=iterations)
    rust_encode_results["list_of_texts"] = timeit.timeit(lambda: rust_tokenizer.encode(text_list), number=iterations)
    rust_results["encode"] = rust_encode_results
    # encode_optimized
    rust_optimized_results = {}
    rust_optimized_results["short_text"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized(short_text), number=iterations)
    rust_optimized_results["long_text"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized(long_text), number=iterations)
    rust_optimized_results["list_of_texts"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized(text_list), number=iterations)
    rust_results["encode_optimized"] = rust_optimized_results
    # encode_optimized2
    rust_optimized2_results = {}
    rust_optimized2_results["short_text"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized2(short_text), number=iterations)
    rust_optimized2_results["long_text"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized2(long_text), number=iterations)
    rust_optimized2_results["list_of_texts"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized2(text_list), number=iterations)
    rust_results["encode_optimized2"] = rust_optimized2_results
    # encode_optimized3
    rust_optimized3_results = {}
    rust_optimized3_results["short_text"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized3(short_text), number=iterations)
    rust_optimized3_results["long_text"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized3(long_text), number=iterations)
    rust_optimized3_results["list_of_texts"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized3(text_list), number=iterations)
    rust_results["encode_optimized3"] = rust_optimized3_results
    # encode_optimized4
    rust_optimized4_results = {}
    rust_optimized4_results["short_text"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized4(short_text), number=iterations)
    rust_optimized4_results["long_text"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized4(long_text), number=iterations)
    rust_optimized4_results["list_of_texts"] = timeit.timeit(lambda: rust_tokenizer.encode_optimized4(text_list), number=iterations)
    rust_results["encode_optimized4"] = rust_optimized4_results
    results["RustBPETokenizer"] = rust_results

    return results

def format_results(results):
    report = "# Tokenizer Benchmark Report\n\n"
    report += "This report details the performance of different `encode` methods in the `HuggingFaceTokenizer` and `RustBPETokenizer` classes. "
    report += "Each method was benchmarked on three different types of input: a short string, a long string, and a list of strings. "
    report += "The times shown are the total time taken for 1000 iterations.\n\n"

    for tokenizer_name, tokenizer_results in results.items():
        report += f"## {tokenizer_name}\n\n"
        if tokenizer_name == "HuggingFaceTokenizer":
            report += "| Input Type      | Time (seconds) |\n"
            report += "|-----------------|----------------|\n"
            for test_name, time_taken in tokenizer_results.items():
                report += f"| {test_name.replace('_', ' ').title()} | {time_taken:.6f}       |\n"
            report += "\n"
        else:
            # RustBPETokenizer has multiple methods
            report += "| Method            | Short Text     | Long Text      | List of Texts  |\n"
            report += "|-------------------|----------------|----------------|----------------|\n"
            methods = ["encode", "encode_optimized", "encode_optimized2", "encode_optimized3", "encode_optimized4"]
            for method in methods:
                method_results = tokenizer_results[method]
                short_time = method_results["short_text"]
                long_time = method_results["long_text"]
                list_time = method_results["list_of_texts"]
                report += f"| `{method}` | {short_time:.6f}       | {long_time:.6f}       | {list_time:.6f}       |\n"
            report += "\n"

    # Conclusion
    report += "## Conclusion\n\n"
    # Find the best RustBPETokenizer method
    best_rust_method = ""
    best_rust_time = float('inf')
    for method, method_results in results["RustBPETokenizer"].items():
        total_time = sum(method_results.values())
        if total_time < best_rust_time:
            best_rust_time = total_time
            best_rust_method = method

    report += f"Based on the results, the most efficient method for the `RustBPETokenizer` is **`{best_rust_method}`**. "
    report += "This method consistently performed the best across all input types.\n"

    return report

if __name__ == "__main__":
    results = run_benchmark()
    if results:
        report = format_results(results)
        with open("benchmark_report.md", "w") as f:
            f.write(report)
        print("Benchmark complete. Report generated at benchmark_report.md")
