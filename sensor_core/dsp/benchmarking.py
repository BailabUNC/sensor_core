import numpy as np
import time
import moving_average_implementations
# Add folder with .so to sys.path
# so_dir = os.path.expanduser('~/Desktop/bailab/sensor_core/sensor_core/dsp')
# sys.path.append(so_dir)


# Now import compiled Cython module
from sensor_core.dsp.moving_average_cython import moving_average_filter
from sensor_core.dsp.moving_average_cython import cython_moving_average_filter_centeredrolling

# Functions already defined: PythonFilter, NumbaFilter, CythonCumSumFilter

def benchmark_filters(sizes, trials=5, window_size=10, pad="min"):
    results = {"Python": [], "Numba": [], "CythonCumSum": [], "Cython": []}
    
    for n in sizes:
        # store trial times
        python_times, numba_times, cython_times, cython2_times = [], [], [], []
        
        for _ in range(trials):
            data = np.random.rand(n)
            
            # PythonFilter
            start = time.perf_counter()
            moving_average_implementations.PythonFilter.moving_average_filter(data, window_size, pad)
            python_times.append(time.perf_counter() - start)
            
            # NumbaFilter (ignore first run warmup JIT)
            moving_average_implementations.NumbaFilter.moving_average_filter(data, window_size, pad)
            start = time.perf_counter()
            moving_average_implementations.NumbaFilter.moving_average_filter(data, window_size, pad)
            numba_times.append(time.perf_counter() - start)
            
            # CythonCumSumFilter
            start = time.perf_counter()
            cython_moving_average_filter_centeredrolling(data, window_size, pad)
            cython_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            moving_average_filter(data, window_size, pad)
            cython2_times.append(time.perf_counter() - start)
        
        # record average for each implementation
        results["Python"].append(np.mean(python_times))
        results["Numba"].append(np.mean(numba_times))
        results["CythonCumSum"].append(np.mean(cython_times))
        results["Cython"].append(np.mean(cython2_times))
    
    return results


if __name__ == "__main__":
    sizes = [10**3, 10**4, 10**5, 10**6]  # data sizes
    results = benchmark_filters(sizes)
    
    # Print results as table
    print("Average runtime over 5 trials (seconds):")
    print(f"{'Size':>10} | {'Python':>10} | {'Numba':>10} | {'CythonCumSum':>12} | {'Cython':>10}")
    print("-" * 50)
    for i, n in enumerate(sizes):
        print(f"{n:>10} | {results['Python'][i]:>10.6f} | {results['Numba'][i]:>10.6f} | {results['CythonCumSum'][i]:>12.6f} | {results['Cython'][i]:>10.6f}")