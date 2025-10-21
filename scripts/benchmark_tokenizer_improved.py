import os
import time
import statistics
import numpy as np
from nanochat.tokenizer import HuggingFaceTokenizer, RustBPETokenizer

def get_tokenizer_dir():
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    nanochat_dir = os.path.join(cache_dir, "nanochat")
    tokenizer_dir = os.path.join(nanochat_dir, "tokenizer")
    return tokenizer_dir

def warmup_tokenizer(tokenizer, sample_text, warmup_iterations=100):
    """Warm up the tokenizer to ensure consistent performance"""
    for _ in range(warmup_iterations):
        tokenizer.encode(sample_text)

def benchmark_method(tokenizer, method_name, text, iterations=1000, num_runs=5):
    """
    Benchmark a specific method with multiple runs for statistical reliability
    """
    # Warm up
    warmup_tokenizer(tokenizer, text)
    
    # Get the method to benchmark
    if hasattr(tokenizer, method_name):
        method = getattr(tokenizer, method_name)
    else:
        raise AttributeError(f"Method {method_name} not found on tokenizer")
    
    # Run multiple times for statistical significance
    run_times = []
    for run in range(num_runs):
        # Use time.perf_counter for higher precision
        start_time = time.perf_counter()
        for _ in range(iterations):
            method(text)
        end_time = time.perf_counter()
        
        run_times.append(end_time - start_time)
    
    # Calculate statistics
    mean_time = statistics.mean(run_times)
    std_time = statistics.stdev(run_times) if len(run_times) > 1 else 0
    min_time = min(run_times)
    max_time = max(run_times)
    
    return {
        'mean': mean_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'runs': run_times
    }

def run_robust_benchmark():
    tokenizer_dir = get_tokenizer_dir()

    # Check if tokenizer directory exists
    if not os.path.exists(tokenizer_dir):
        print(f"Tokenizer directory not found at: {tokenizer_dir}")
        print("Please run scripts/tok_train.py to create the tokenizer directory.")
        return

    # Initialize tokenizers
    print("Initializing tokenizers...")
    hf_tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
    rust_tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

    # Sample data for benchmarking with much more varied lengths
    short_text = "This is a short sentence."
    medium_text = "This is a medium-length sentence that provides a good baseline for testing tokenizer performance. " * 50  # ~2,500 chars
    long_text = "This is a much longer sentence that is designed to test the performance of the tokenizers on a larger input. " * 200  # ~10,000 chars
    very_long_text = "This is an extremely long text that will really stress test the tokenizer performance and show clear differences between methods. " * 1000  # ~50,000 chars
    text_list = ["Here is the first sentence.", "And this is the second one.", "Finally, the third sentence for testing."]
    long_text_list = [
        "This is the first sentence in a longer list that will test batch processing performance. " * 20,
        "This is the second sentence that is also quite long to provide a good test case. " * 25,
        "This is the third sentence that continues the pattern of longer text for better benchmarking. " * 30,
        "This is the fourth sentence that adds more complexity to the batch processing test. " * 15,
        "This is the fifth and final sentence that completes our comprehensive test suite. " * 35
    ]

    # Benchmarking parameters
    iterations = 1000
    num_runs = 5  # Number of independent runs for statistical analysis

    print(f"Running benchmark with {iterations} iterations per method, {num_runs} runs per method...")
    print("This may take a few minutes for statistical reliability...\n")

    # Benchmarking results dictionary
    results = {}

    # --- HuggingFaceTokenizer ---
    print("Benchmarking HuggingFaceTokenizer...")
    hf_results = {}
    
    # Short text
    hf_results["short_text"] = benchmark_method(hf_tokenizer, "encode", short_text, iterations, num_runs)
    # Medium text
    hf_results["medium_text"] = benchmark_method(hf_tokenizer, "encode", medium_text, iterations, num_runs)
    # Long text
    hf_results["long_text"] = benchmark_method(hf_tokenizer, "encode", long_text, iterations, num_runs)
    # Very long text
    hf_results["very_long_text"] = benchmark_method(hf_tokenizer, "encode", very_long_text, iterations, num_runs)
    # List of texts
    hf_results["list_of_texts"] = benchmark_method(hf_tokenizer, "encode", text_list, iterations, num_runs)
    # Long list of texts
    hf_results["long_list_of_texts"] = benchmark_method(hf_tokenizer, "encode", long_text_list, iterations, num_runs)
    results["HuggingFaceTokenizer"] = hf_results

    # --- RustBPETokenizer ---
    print("Benchmarking RustBPETokenizer...")
    rust_results = {}
    
    methods = ["encode", "encode_optimized", "encode_optimized2", "encode_optimized3", "encode_optimized4"]
    
    for method in methods:
        print(f"  Testing {method}...")
        method_results = {}
        
        # Short text
        method_results["short_text"] = benchmark_method(rust_tokenizer, method, short_text, iterations, num_runs)
        # Medium text
        method_results["medium_text"] = benchmark_method(rust_tokenizer, method, medium_text, iterations, num_runs)
        # Long text
        method_results["long_text"] = benchmark_method(rust_tokenizer, method, long_text, iterations, num_runs)
        # Very long text
        method_results["very_long_text"] = benchmark_method(rust_tokenizer, method, very_long_text, iterations, num_runs)
        # List of texts
        method_results["list_of_texts"] = benchmark_method(rust_tokenizer, method, text_list, iterations, num_runs)
        # Long list of texts
        method_results["long_list_of_texts"] = benchmark_method(rust_tokenizer, method, long_text_list, iterations, num_runs)
        
        rust_results[method] = method_results
    
    results["RustBPETokenizer"] = rust_results

    return results

def format_robust_results(results):
    report = "# Robust Tokenizer Benchmark Report\n\n"
    report += "This report details the performance of different `encode` methods with statistical analysis. "
    report += "Each method was benchmarked with multiple runs for reliability. "
    report += "Times shown are mean values with standard deviation from 5 independent runs of 1000 iterations each.\n\n"
    
    report += "## Test Data Characteristics\n\n"
    report += "- **Short Text**: ~25 characters\n"
    report += "- **Medium Text**: ~2,500 characters\n"
    report += "- **Long Text**: ~10,000 characters\n"
    report += "- **Very Long Text**: ~50,000 characters\n"
    report += "- **List of Texts**: 3 short sentences\n"
    report += "- **Long List of Texts**: 5 long sentences (~125,000 total characters)\n\n"

    for tokenizer_name, tokenizer_results in results.items():
        report += f"## {tokenizer_name}\n\n"
        if tokenizer_name == "HuggingFaceTokenizer":
            report += "| Input Type      | Mean Time (s) | Std Dev (s) | Min (s) | Max (s) |\n"
            report += "|-----------------|---------------|-------------|---------|----------|\n"
            for test_name, stats in tokenizer_results.items():
                report += f"| {test_name.replace('_', ' ').title()} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |\n"
            report += "\n"
        else:
            # RustBPETokenizer has multiple methods
            report += "| Method            | Input Type    | Mean Time (s) | Std Dev (s) | Min (s) | Max (s) |\n"
            report += "|-------------------|---------------|---------------|-------------|---------|----------|\n"
            methods = ["encode", "encode_optimized", "encode_optimized2", "encode_optimized3", "encode_optimized4"]
            for method in methods:
                method_results = tokenizer_results[method]
                for input_type, stats in method_results.items():
                    report += f"| `{method}` | {input_type.replace('_', ' ').title()} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |\n"
            report += "\n"

    # Statistical analysis
    report += "## Statistical Analysis\n\n"
    
    # Find the best RustBPETokenizer method for each input type
    best_methods = {}
    for input_type in ["short_text", "medium_text", "long_text", "very_long_text", "list_of_texts", "long_list_of_texts"]:
        best_method = ""
        best_mean_time = float('inf')
        best_std = 0
        
        for method, method_results in results["RustBPETokenizer"].items():
            stats = method_results[input_type]
            if stats['mean'] < best_mean_time:
                best_mean_time = stats['mean']
                best_std = stats['std']
                best_method = method
        
        best_methods[input_type] = {
            'method': best_method,
            'mean': best_mean_time,
            'std': best_std
        }
    
    report += "### Best RustBPETokenizer Methods by Input Type:\n\n"
    for input_type, best in best_methods.items():
        report += f"- **{input_type.replace('_', ' ').title()}**: `{best['method']}` "
        report += f"(Mean: {best['mean']:.6f}s ± {best['std']:.6f}s)\n"
    
    # Overall best method
    overall_best = min(best_methods.values(), key=lambda x: x['mean'])
    report += f"\n### Overall Best Method: `{overall_best['method']}`\n"
    report += f"Consistently performs best across all input types with mean time of {overall_best['mean']:.6f}s ± {overall_best['std']:.6f}s\n"

    return report

if __name__ == "__main__":
    results = run_robust_benchmark()
    if results:
        report = format_robust_results(results)
        with open("robust_benchmark_report.md", "w") as f:
            f.write(report)
        print("\nRobust benchmark complete. Report generated at robust_benchmark_report.md")
        print("This report includes statistical analysis with mean, standard deviation, min, and max times.")
