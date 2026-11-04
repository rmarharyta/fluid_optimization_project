using System.Numerics;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

class Program
{
    static void Main(string[] args)
    {
        // Compare the performance of different LBM implementations
        BenchmarkRunner.Run<LbmPerformanceTests>();
    }
}

// Iteration 1: Naive Object-Oriented Approach (Baseline)
public class Cell
{
    //Each cell holds a reference to its own array of 8 state components
    public float[] F { get; set; } = new float[8];
}

public class LbmBaseline
{
    private readonly Cell[,] _grid;
    private readonly int _width;
    private readonly int _height;
    private readonly float _gamma = 0.85f;
    private readonly float _w = 0.125f;

    public LbmBaseline(int width, int height)
    {
        _width = width;
        _height = height;
        _grid = new Cell[width, height];
        for (int y = 0; y < _height; y++)
            for (int x = 0; x < _width; x++)
                _grid[x, y] = new Cell();
    }

    public void ComputeKernel()
    {
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                Cell cell = _grid[x, y];

                //1. Reduction (Sum Gathering)
                float phi = 0;
                for (int i = 0; i < 8; i++)
                    phi += cell.F[i];

                // 2. Local State Update
                for (int i = 0; i < 8; i++)
                {
                    cell.F[i] = cell.F[i] * (1f - _gamma) + _w * phi * _gamma;
                }
            }
        }
    }
}

//Iteration 2: Memory Optimization (Linear Layout & Zero-Allocation)
public class LbmMemoryOptimized
{
    private readonly float[] _f;
    private readonly int _width;
    private readonly int _height;
    private readonly int _gridSize;
    private readonly float _gamma = 0.85f;
    private readonly float _w = 0.125f;

    public LbmMemoryOptimized(int width, int height)
    {
        _width = width;
        _height = height;
        _gridSize = width * height;
        // A single contiguous array where components are stored in long slices (SoA)
        _f = new float[8 * _gridSize];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int GetIndex(int direction, int nodeIdx) => direction * _gridSize + nodeIdx;

    public void ComputeKernel()
    {
        // Create a Span for safe and fast access without heap allocations
        Span<float> fSpan = _f.AsSpan();

        for (int idx = 0; idx < _gridSize; idx++)
        {
            float phi = 0;
            for (int i = 0; i < 8; i++)
                phi += fSpan[GetIndex(i, idx)];

            float phiGamma = phi * _gamma;
            float oneMinusGamma = 1f - _gamma;

            for (int i = 0; i < 8; i++)
            {
                int fIdx = GetIndex(i, idx);
                fSpan[fIdx] = fSpan[fIdx] * oneMinusGamma + _w * phiGamma;
            }
        }
    }
}

//Iteration 3: SIMD Hardware Intrinsics (Vectorization)
public class LbmSimd
{
    private readonly float[] _f;
    private readonly int _gridSize;
    private readonly float _gamma = 0.85f;
    private readonly Vector<float> _wVec = new Vector<float>(0.125f);

    public LbmSimd(int width, int height)
    {
        _gridSize = width * height;
        _f = new float[8 * _gridSize];
    }

    public void ComputeKernel()
    {
        Span<float> fSpan = _f.AsSpan();
        int vSize = Vector<float>.Count; // 8 for AVX2 (256 bits) or 16 for AVX-512
        int idx = 0;

        Vector<float> vGamma = new Vector<float>(_gamma);
        Vector<float> vOneMinusGamma = new Vector<float>(1f - _gamma);

        // Main SIMD loop: process batches of nodes in parallel at the hardware level
        for (; idx <= _gridSize - vSize; idx += vSize)
        {
            // Load data for 8/16 nodes for each of the 8 directions directly into CPU registers
            Vector<float> v0 = new Vector<float>(fSpan.Slice(0 * _gridSize + idx, vSize));
            Vector<float> v1 = new Vector<float>(fSpan.Slice(1 * _gridSize + idx, vSize));
            Vector<float> v2 = new Vector<float>(fSpan.Slice(2 * _gridSize + idx, vSize));
            Vector<float> v3 = new Vector<float>(fSpan.Slice(3 * _gridSize + idx, vSize));
            Vector<float> v4 = new Vector<float>(fSpan.Slice(4 * _gridSize + idx, vSize));
            Vector<float> v5 = new Vector<float>(fSpan.Slice(5 * _gridSize + idx, vSize));
            Vector<float> v6 = new Vector<float>(fSpan.Slice(6 * _gridSize + idx, vSize));
            Vector<float> v7 = new Vector<float>(fSpan.Slice(7 * _gridSize + idx, vSize));

            //Mathematical vector reduction
            Vector<float> vPhi = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
            Vector<float> vPhiGamma = vPhi * vGamma;

            // Vectorized update and write-back to memory
            (v0 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(0 * _gridSize + idx, vSize));
            (v1 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(1 * _gridSize + idx, vSize));
            (v2 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(2 * _gridSize + idx, vSize));
            (v3 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(3 * _gridSize + idx, vSize));
            (v4 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(4 * _gridSize + idx, vSize));
            (v5 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(5 * _gridSize + idx, vSize));
            (v6 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(6 * _gridSize + idx, vSize));
            (v7 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(7 * _gridSize + idx, vSize));
        }

        // Handle the remaining nodes that do not fit into a full vector
        for (; idx < _gridSize; idx++)
        {
            float phi = 0;
            for (int i = 0; i < 8; i++) phi += fSpan[i * _gridSize + idx];
            for (int i = 0; i < 8; i++)
            {
                int fIdx = i * _gridSize + idx;
                fSpan[fIdx] = fSpan[fIdx] * (1f - _gamma) + 0.125f * phi * _gamma;
            }
        }
    }
}

//Iteration 4: Parallel SIMD Pipeline (Multi-Threading + SIMD)
public class LbmParallelSimd
{
    private readonly float[] _f;
    private readonly int _width;
    private readonly int _height;
    private readonly int _gridSize;
    private readonly float _gamma = 0.85f;
    private readonly Vector<float> _wVec = new Vector<float>(0.125f);

    public LbmParallelSimd(int width, int height)
    {
        _width = width;
        _height = height;
        _gridSize = width * height;
        _f = new float[8 * _gridSize];
    }

    public void ComputeKernel()
    {
        // UnParallelized SIMD computation across rows of the grid
        Parallel.For(0, _height, y =>
        {
            Span<float> fSpan = _f.AsSpan();
            int vSize = Vector<float>.Count;
            int rowOffset = y * _width;

            Vector<float> vGamma = new Vector<float>(_gamma);
            Vector<float> vOneMinusGamma = new Vector<float>(1f - _gamma);

            int x = 0;
            // Each CPU core performs SIMD computations on its assigned row
            for (; x <= _width - vSize; x += vSize)
            {
                int idx = rowOffset + x;

                Vector<float> v0 = new Vector<float>(fSpan.Slice(0 * _gridSize + idx, vSize));
                Vector<float> v1 = new Vector<float>(fSpan.Slice(1 * _gridSize + idx, vSize));
                Vector<float> v2 = new Vector<float>(fSpan.Slice(2 * _gridSize + idx, vSize));
                Vector<float> v3 = new Vector<float>(fSpan.Slice(3 * _gridSize + idx, vSize));
                Vector<float> v4 = new Vector<float>(fSpan.Slice(4 * _gridSize + idx, vSize));
                Vector<float> v5 = new Vector<float>(fSpan.Slice(5 * _gridSize + idx, vSize));
                Vector<float> v6 = new Vector<float>(fSpan.Slice(6 * _gridSize + idx, vSize));
                Vector<float> v7 = new Vector<float>(fSpan.Slice(7 * _gridSize + idx, vSize));

                Vector<float> vPhi = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
                Vector<float> vPhiGamma = vPhi * vGamma;

                (v0 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(0 * _gridSize + idx, vSize));
                (v1 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(1 * _gridSize + idx, vSize));
                (v2 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(2 * _gridSize + idx, vSize));
                (v3 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(3 * _gridSize + idx, vSize));
                (v4 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(4 * _gridSize + idx, vSize));
                (v5 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(5 * _gridSize + idx, vSize));
                (v6 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(6 * _gridSize + idx, vSize));
                (v7 * vOneMinusGamma + _wVec * vPhiGamma).CopyTo(fSpan.Slice(7 * _gridSize + idx, vSize));
            }

            // Scalar tail for the remaining nodes in the row
            for (; x < _width; x++)
            {
                int idx = rowOffset + x;
                float phi = 0;
                for (int i = 0; i < 8; i++) phi += fSpan[i * _gridSize + idx];
                for (int i = 0; i < 8; i++)
                {
                    int fIdx = i * _gridSize + idx;
                    fSpan[fIdx] = fSpan[fIdx] * (1f - _gamma) + 0.125f * phi * _gamma;
                }
            }
        });
    }
}

[MemoryDiagnoser]
public class LbmPerformanceTests
{
    private const int Width = 512;
    private const int Height = 512;
    private LbmBaseline _baseline = null!;
    private LbmMemoryOptimized _memoryOptimized = null!;
    private LbmSimd _simd = null!;
    private LbmParallelSimd _parallelSimd = null!;
    [GlobalSetup]
    public void Setup()
    {
        _baseline = new LbmBaseline(Width, Height);
        _memoryOptimized = new LbmMemoryOptimized(Width, Height);
        _simd = new LbmSimd(Width, Height);
        _parallelSimd = new LbmParallelSimd(Width, Height);
    }
    [Benchmark(Baseline = true)]
    public void Iteration_1_Baseline() => _baseline.ComputeKernel();
    [Benchmark]
    public void Iteration_2_MemoryOptimized() => _memoryOptimized.ComputeKernel();
    [Benchmark]
    public void Iteration_3_Simd() => _simd.ComputeKernel();
    [Benchmark]
    public void Iteration_4_ParallelSimd() => _parallelSimd.ComputeKernel();
}