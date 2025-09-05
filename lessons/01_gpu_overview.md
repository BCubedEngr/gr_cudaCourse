# GPU Overview

You likely already have a pretty good idea of what a GPU is why it can be much faster than a CPU.  Many of the top line numbers that you see relate to ML tasks and aren't quite as applicable to DSP/Comms problems.  Let's go over some of the numbers and parse out what is there.

### GPU Core Types
#### Tensor Cores
Performs Matrix Multiply-Accumulate -> `D = A*B + C`  This is a very common part of machine learning workflows.  The Tensor Cores make quick work of matrix operations.  Sadly, most DSP/Comms operations cannot be efficiently represented using matrix operations.  These cores don't tend to help us much.
#### Ray Tracing Cores
Searches for intersections on rays.  Useful for video games, not very useful for DSP/Comms operations.
#### Cuda Core
Like a light weight general purpose CPU.  These are the cores that are most helpful when doing our work.

### Example Processors
#### CPUs
https://www.intel.com/content/www/us/en/products/sku/236783/intel-core-i7-processor-14700k-33m-cache-up-to-5-60-ghz/specifications.html
https://www.amd.com/en/products/processors/workstations/ryzen-threadripper/9000-wx-series/amd-ryzen-threadripper-pro-9995wx.html
https://amperecomputing.com/briefs/ampereone-m-product-brief

| Name                                | Mode  | Clock Hz | Cores | Threads | Power | Approx FLOPS | GFLOPS/W |
|------------------------------------------------------------------------------------------------------------|
| Intel i7-14700K                     | Base  | 2.5 GHz  | 20    | 28      | 125 W |  560 G       | 4.48     |
| Intel i7-14700K                     | Turbo | 5.6 GHz  | 20    | 28      | 253 W | 1254 G       | 4.95     |
| AMD Ryzen™ Threadripper™ PRO 9995WX | Base  | 2.5 GHz  | 96    | 192     | ???   | 3840 G       | ???      |
| AMD Ryzen™ Threadripper™ PRO 9995WX | Turbo | 5.4 GHz  | 96    | 192     | 350 W | 8294 G       | 23.70    |
| AmpereOne® A192-32M                 | Base  | 3.2 GHz  | 192   | 192     | 348 W | 2457 G       | 7.06     |

Back of the envelop formula for FLOPS: `clock_speed * threads * simd_floats / 2`.  Note that SIMD instructions generally take longer than their scalar counterparts hence the divide by 2.  Also Hyper threaded cores frequently don't live up to their full potential, so these are over estimates.

#### GPUs
https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
https://www.nvidia.com/en-us/data-center/l4/

| Name                                 | Clock Hz | Cuda Cores | Power | Approx FLOPS | GFLOPS/W |
|------------------------------------------------------------------------------------------------|
| Jetson Orin Nano Super Developer Kit | 1.02 GHz | 1024       | 25 W  | 1044 G       | 41.76    |
| Nvidia L4                            | 2.04 GHz | 7680       | 72 W  | 15667 G      | 217.60   |

Simple Formula for FLOPS: `clock_speed * cuda_cores`.

### How to build and run this tutorial?
To run the code, you will need access to an NVIDIA GPU.  The code has been tested on Linux.  It will likely work on MAC or Windows systems, but this hasn't been verified.  The `Jetson Orin Nano Super Developer Kit` costs $250 and a cheap way to get access to a fairly powerful GPU.  the kit comes ready to build and run GPU applications.

Building the code only requires access to the `nvcc` compiler.  Note that the version of `nvcc` used needs to be supported by the NVIDIA driver on your system if you are running the code.
This repository contains a `Dockerfile` that builds a container that can be used for building the examples in this tutorial.  Alternatively, several of the initial examples can be built
online using `https://godbolt.org/noscript/cuda`.

### Core Concepts
#### Kernel
A kernel is code that compiles to run on the GPU.  Here is a simple example:
```
__global__ void add1(float* x, int length) {
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        x[i] += 1.0;
    }
}

void run_kernel(float* x, int length) {
    // Each thread processes 8 points
    add1<<<1, length/8>>>(x, length);
    cudaDeviceSynchronize();
}

int main() {
    float* x;
    cudaMalloc(&x, 100*sizeof(float));
    run_kernel(x, 100);
    cudaFree(x);
}

```

##### Function Qualifiers
Cuda uses 3 function qualifiers to denote where a function runs and where it can be called.  `__global__` functions run on the GPU and are callable from the host code.  `__device__` functions run on the GPU but can only be called directly from the GPU.  (Think of it as GPU helper functions.)  `__host__` functions run on the CPU and are only callable from the CPU.  These are the normal c++ functions that you already know how to write and use.  If a qualifier isn't added to our function, then it is assumed to have the `__host__` qualifier.

##### Kernel Launch Parameters
Kernels are launched with extra parameters enclosed between `<<<gridDim, blockDim, [shared_mem_size], [stream]>>>`.  CUDA kernels are launched in a multi dimensional grid.  This can make things like a matrix multiply easier because we can align the grid size to the matrix size.  The `gridDim` and `blockDim` are triples representing `x,y,z` dimensions.  However, it a single value is entered, then that becomes the x value and the y and z values default to 1.

We divide the problem into a set of blocks and then a number of threads work on each block.  The `gridDim` is the number of blocks in the problem.  The `blockDim` is the number of threads per block.  Note that there is a limit to the number of threads per block (GPU dependent), but no limit on the number of blocks for a particular kernel launch.

Some kernels use shared memory between threads.  In our example, we don't need any memory and can leave it off.  We will discuss the `stream` parameter later.  It can be ignored for now.

##### Built In Variables
Cuda provides several variables in device code that are automatically populated when the kernel is called.  This can be somewhat confusing because the variables aren't declared anywhere.  There are two examples in the above kernel - `threadIdx.x` and `blockDim.x`.  Since we are treating this as a 1-dimensional kernel, we don't need to worry about variables like `threadIdx.y` or `blockDim.z`.

All of the threads in a block execute the exact same code.  They only differ in places where these special variables are used.  In the above code: `for (int i = threadIdx.x; i < length; i += blockDim.x)` means that each thread starts processing the array on the index associated with its `threadIdx` and steps by the `blockDim`.  There is a problem with the above kernel.  What happens if we call it with multiple blocks?

We will have a problem with thread pairs such as - `blockIdx.x = 0, threadIdx.x = 0` and `blockIdx.x = 1, threadIdx.x = 0`.  These will both operate on the same data.  We need to update our kernel to work with multiple blocks.

```
__global__ void add1(float* x, int length) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x) {
        x[i] += 1.0;
    }
}

void run_kernel(float* x, int length) {
    // Each thread still processes 8 points but with 2 blocks
    add1<<<2, length/16>>>(x, length);
    cudaDeviceSynchronize();
}
```

The above code introduces one more built in variable - `gridDim.x`.  This represents the number of blocks.  So in the code `gridDim.x = 2`, `blockDim.x = 100/16 = 6`, `threadIdx.x` will be a number between 0 and 5, and `blockIdx.x` will be a number between 0 and 1.

##### Warps
One of the ways that GPUs are able to work so efficiently is by having blocks of threads execute the same instructions on the same "block" of memory.  These blocks are called `warps`.  You may have noticed that it we could have written the add1 kernel to have each thread operate on a contiguous chunk of memory.  (Going back to the first example code for simplicity)

```
__global__ void add1(float* x, int length) {
    int start = threadIdx.x * blockDim.x;
    int end = min((threadIdx.x + 1) * blockDim.x, length);
    for (int i = start; i < end; i++) {
        x[i] += 1.0;
    }
}
```

This is valid code and will compile, but the performance will in general be much worse.  Each thread executes the same code, but the memory is much more spread out inside of each warp.  Kernels threads want to work on interleaved data rather than contiguous memory.

##### Streams
Streams are the primary method that CUDA uses to handle concurrency between different kernels.  We can have a single CPU process that wants to run several kernels in order and wait for the GPU to finish.  In other cases the kernels are operating on independent data and we want them to execute simultaneously.  Streams can be thought of as ordered queues of kernels to execute.  Each kernel in a stream executes in order waiting for the previous kernel to finish before starting the next kernel.  Kernels in different streams operate independently.  As the GPU has capacity it will schedule blocks from each stream with attempting any inter-stream synchronization.

The `default stream` is a special stream that executes kernels launched without a stream parameter `<<<gridDim, blockDim, [shared_mem_size], [stream]>>>` or with a stream parameter of 0.  The default stream does synchronize with/block other streams.  Outside of simple toy examples, we will never use the default stream.  If all of our kernels properly use streams and just 1 uses the default stream it can have a massive impact on performance.

##### Synchronization
GPUs are able to do a lot of work very quickly, but we need to move data to the GPU and return the result to CPU in order for the work to be useful.  In a poorly designed program, there may be much more time spent moving data to and from the GPU then time spent acutally processing the data on the GPU.  A well written program will do multiple kernel operations at the same time.  At some point, we will need to know when a kernel has completed running and data can be safely accessed.

```
// These call send the kernel to the GPU, but don't wait for the kernels to complete.
cudaMemcpyAsync(&gpu_dst, &host_src, N);
my_kernel<<<1,100>>>(gpu_dst);
cudaMemcpyAsync(&host_dst, &gpu_dst, N);
// Before synchronize call, pervious kernels are in an unknown state of completion.  We can access host_dst but with non-deterministic results
cudaDeviceSynchronize();
// Now we can access host_dst safely.

```

We can synchronize to the default stream using `cudaDeviceSynchronize` and to a specific stream using `cudaStreamSynchronize`.

##### Memory Allocation
GPUs have their own memory and can't generally access host system memory and the host can't generally access GPU memory - attempt to access the memory from the wrong location will cause your program to crash.

CUDA provides 3 kinds of memory that the GPU can access.

- device: GPU memory that is only accessible to the from `__global__` and `__device__` functions.
- unified: Memory that is accessible from the host and the GPU.  It takes care of transferring the correct data back and forth as needed.
- host pinned: Host memory that can be directly accessed from the GPU - think DMA 

#### CCCL
As we learn to write kernels, frequently the best kernel is one that you don't have to write.  The `CUDA Core Compute Libraries` are a combination of several projects designed to make efficient use of the GPU simple: `https://github.com/NVIDIA/cccl`.  There are functions and classes that look much more like standard c++ and come with highly optimized implementations.