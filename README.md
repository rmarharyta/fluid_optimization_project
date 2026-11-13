# High-Performance LBM D2Q9 Fluid Simulation in .NET


## Overview
This repository contains a high-performance Computational Fluid Dynamics (CFD) solver based on the **Lattice Boltzmann Method (LBM D2Q9)**, implemented in modern **C# / .NET**. 

The project demonstrates that modern managed .NET platforms can achieve near-native computational speeds (comparable to C++/Fortran) by leveraging low-level memory structures (`Span<T>`), hardware SIMD intrinsics (`Vector<T>`), and multi-threading (`Parallel.For`).



## Mathematical Core

The solver models 2D incompressible fluid flow using the discrete Boltzmann equation with the single-relaxation-time **BGK** collision operator:

$$f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) = f_i(\mathbf{x}, t) - \frac{1}{\tau} \left( f_i(\mathbf{x}, t) - f_i^{eq}(\mathbf{x}, t) \right), \quad i = 0, \dots, 8$$

Where the local equilibrium distribution $f_i^{eq}$ and macroscopic density ($\rho$) and velocity ($\mathbf{u}$) are computed as:

$$f_i^{eq}(\rho, \mathbf{u}) = w_i \rho \left[ 1 + 3(\mathbf{c}_i \cdot \mathbf{u}) + \frac{9}{2}(\mathbf{c}_i \cdot \mathbf{u})^2 - \frac{3}{2}(\mathbf{u} \cdot \mathbf{u}) \right]$$

$$\rho = \sum_{i=0}^{8} f_i, \quad \mathbf{u} = \frac{1}{\rho} \sum_{i=0}^{8} f_i \mathbf{c}_i$$



## Optimization Stages

1. **Baseline (`Iteration_1_Baseline`)**: Standard 3D arrays (`double[NX, NY, 9]`) with nested loops. Suffers from CLR bounds-checking overhead and CPU cache misses during the streaming step.
2. **Memory-Optimized (`Iteration_2_MemoryOptimized`)**: Flattened 1D array wrapped in `Span<T>` to achieve Zero-Allocation memory access. *Note: Exhibited a temporary slowdown due to scalar 3D-to-1D index calculation overhead.*
3. **SIMD Accelerated (`Iteration_3_Simd`)**: Vectorized lattice operations using `System.Numerics.Vector<double>`, loading node vectors directly into wide CPU registers (AVX2) and eliminating index bottlenecks.
4. **Parallel + SIMD (`Iteration_4_ParallelSimd`)**: Combined hardware SIMD intrinsics with multi-threaded domain decomposition (`Parallel.For`) across all CPU cores.



## Benchmark Results

Evaluated via **BenchmarkDotNet** (Release mode) measuring mean execution time per simulation step:

| Method | Mean Time | StdDev | Median | Allocated | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Iteration_1_Baseline` | 14,216.2 μs | 2,126.22 μs | 13,931.4 μs | 0 B | **1.00x** |
| `Iteration_2_MemoryOptimized` | 18,925.1 μs | 1,860.79 μs | 17,447.2 μs | 0 B | **0.75x** |
| `Iteration_3_Simd` | 1,584.3 μs | 182.87 μs | 1,586.5 μs | 0 B | **8.97x** |
| `Iteration_4_ParallelSimd` | **444.5 μs** | **18.92 μs** | **445.1 μs** | **3,569 B** | **31.98x** |

> **Result:** Combining SIMD vectorization with parallel execution yielded a **31.98x speedup** over the baseline with zero GC pressure inside the simulation loop.



##  Usage

### Prerequisites
* [.NET 8.0 SDK](https://microsoft.comdownload) or higher.
* CPU with AVX2 / SIMD instruction support.

### Run Benchmarks
```bash
git clone https://github.com
cd lbm-d2q9-dotnet
dotnet run -c Release
```



## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
