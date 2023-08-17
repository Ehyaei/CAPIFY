from src.run_benchmarks import run_benchmark

for i in range(100):
    run_benchmark(i*100)
    print("=====================================================")
    print(f"Finished {i}th benchmark")
    print("=====================================================")