## Simple Frequency Shift Kernel
We are going to start our CUDA dsp journey with a simple kernel - a frequency tuner.  This takes in an input signal and just multiplies by a 
complex exponential `y[n] = x[n] * e^(2*pi*n*f)`.

### The Function Signature
```
__global__ void tuner_kernel(cuda::std::complex<float>* out, const cuda::std::complex<float>* in, size_t size, float freq, float phase) {

}
```

Let's break apart the various pieces.

#### __global__ void
All cuda kernels must start with `__global__ void`.  The `__global__` tells the cuda compiler that this code runs on the device but is callable
from the host.

Since kernels are called asynchronously, we can't return a value in their functions directly.  Mutable values have to be passed in as function
arguments.  When the kernel is completed we can access the value (possibly needing to copy the data back to the host).

#### cuda::std::complex<float>
It can be tricky to use structures and classes on the GPU.  Any functions that are callable on the GPU has to have the `__device__` qualifier.
But the standard template library (STL) wasn't written with cuda in mind.  The following code would fail to compile:

```
__global__ void bad_kernel(std::complex<float>* out, const std::complex<float>* in) {
    float in_real = in[threadIdx.x].real();  // real() isn't compiled for device functions.
    out[threadIdx.x] = in[threadIdx.x] * in_real; // operator* and operator= also don't work
}
```

Essentially we can't do anything useful with `std::complex` values inside of a kernel so cuda defines `cuda::std::complex` which works for
both host and device code.

### Filling out the code
```
__global__ void tuner_kernel(cuda::std::complex<float>* out, const cuda::std::complex<float>* in, size_t size, float freq, float phase) {
    for (size_t i = get_start(); i < size; i =+ get_stride()) {
        // Compute input arg
        auto carg = freq * i + phase;

        // Produce complex eponential
        cuda::std::complex<float> _out;
        float* val = reinterpret_cast<float*>(&_out);
        sincosf(carg, val + 1, val);

        // Generate output
        out[i] = in[i] * _out;
    }
}
```
Again, let's break down the pieces

#### Helper Functions
Generally, with signal processing we are working with 1d data.  There is some boiler plate code that can be annoying to copy and paste all over.
In `cuda_util.cuh` we create some helper functions.

```
__device__ __forceinline__ int get_start() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int get_stride() {
    return blockDim.x * gridDim.x;
}
```

These blocks help us iterate over the data in the way that is most efficient for the warp structure of the GPU.  Notice that these functions have
the `__device__` qualifier.  This means that these can be called from the GPU.  The `__forceinline__` ensures that the code is inserted directly
into the calling site, rather than via a function call.  This can improve performance of really small functions.

#### Complex Exponential Calculation
```
        // Produce complex eponential
        cuda::std::complex<float> _out;
        float* val = reinterpret_cast<float*>(&_out);
        sincosf(carg, val + 1, val);

```

In this simple case, we are just going to directly compute sin and cos directly in our calls.  These are relatively expensive calls but they
are simple.  CUDA provides the `sincosf` function which calculates sin and cos at the same time.  We have to reverse the order of the output
because our complex exponential is ordered `cos + 1j*sin`.

There are some efficiencies we can gain by lifting the trig functions out of our main loop that will be explored in a challenge exercise.

#### Produce Output
```
        // Generate output
        out[i] = in[i] * _out;
```

Since we are using the `cuda::std::complex<float>` class we can use `operator*` and `operator=` without issue.

#### Wrapper Function
There are some potential issues with the above kernel.  The biggest is how are we keeping track of the phase in between kernel calls?
Also what are we doing to prevent the phase from becoming large.  If we pass large values into sincosf, we won't have the precision to
get accurate results.  We write a wrapper function that takes care of this for us.

```
float vector_tune(complex<float>* out, size_t out_size,
                  const complex<float>* in, size_t in_size,
                  float freq,
                  float phase,
                  const kernel_params& params,
                  CudaStream& stream) {
    if (in_size != out_size) {
        throw runtime_error("In size and out size don't match");
    }
    // Convert to gpu pointer types
    cuda::std::complex<float>* g_out = reinterpret_cast<cuda::std::complex<float>* >(out);
    const cuda::std::complex<float>* g_in =  reinterpret_cast<const cuda::std::complex<float>* >(in);
    tuner_kernel<<<params.grid_size, params.block_size, 0, stream.get()>>>(g_out, g_in, out_size, freq, phase);
    return std::fmod(freq * in_size + phase, 2*M_PI);
}
```

Note that we pass in `std::complex<float>` instead of `cuda::std::complex<float>` and then cast the value to the type the GPU needs.
This makes it easier to work with and call from existing code.

This runs the kernel and then updates the phase for the next call.  Note that it calls `fmod` which prevents the phase from growing
too large.


## Reductions - Sum Kernel
Kernels that output less data then they put in can be tricky.  Each thread runs independently.  What is wrong with the following kernel?
```
__global__ sum_bad(int* result, int* input, int length) {
    for (int i = get_start(); i < length; i += get_stride())
        *result += input[i];
}
```

- What value does result start out with?
- What happens if two threads try to write to result at the same time?
- What happens if one thread reads result while another one writes it?


### Correct Version #1
```
__global__ sum_kernel(int* result, int* input, int length) {
    // Assume that result is initialized to zero before this kernel using cudaMemset
    int index = get_start();
    int temp = input[index];
    input += get_stride();
    for (int i = input; i < length; i += get_stride()) {
        temp += input[i];
    }

    // Combine together
    atomicAdd(result, temp);
}
```

Why wouldn't we do a atomicAdd inside of our loop?

### Correct Version #2
```
__global__ sum_shared_mem(int* result, int* input, int length) {
    // Size of shared memory set when kernel is launched
    extern __shared__ int sdata[];
    
    int index = get_start()
    sdata[threadIdx.x] = input[index];
    input += get_stride();
    for (int i = input; i < length; i += get_stride()) {
        sdata[threadIdx.x] += input[i];
    }
    __syncthreads();
    // Reduce the shared memory down to the result
    for (int i = blockDim.x/2; i > 0; i /= 2) {
        if (threadIdx.x < i)
            sdata[i] += sdata[i + blockDim.x/2];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Assume that result is initialized to zero before this kernel using cudaMemset
        atomicAdd(result, sdata[0]);
    }
}
```

#### Shared Memory
When we launch a kernel, one of the parameters is the amount of shared memory needed.  Shared memory is allocated for each block
so the actual amount of memory used will generally be much larger than the number we put in.

For example, if we call the above with `find_max_shared_mem<<<1, 256, 256*sizeof(int)>>>`.  This will process with 1 block, 
getting 1024 bytes of shared memory.

```
    extern __shared__ int sdata[];
```

The amount of shared memory needed is different on every kernel casll so we declare `sdata []` and it will use the correct size for
the kernel parameter used.

#### __syncthreads()
Sometimes we need to make sure that each thread in a block has finished working before we move on to the next line of code.  The `__syncthreads()`
function does this.  It is important to note that it doesn't sync across blocks, just threads within a block.

#### Reduction Loop
```
    __syncthreads();
    // Reduce the shared memory down to the result
    for (int i = blockDim.x/2; i > 0; i /= 2) {
        if (threadIdx.x < i)
            sdata[i] += sdata[i + blockDim.x/2];
        __syncthreads();
    }
```

We now have the max values for each of the threads.  We need to combine those values into a single value.   We do this over many iterations, in each
iteration reducing the amount of data to process by a factor of 2.  

Why do we need to call `__syncthreads` at the end of each loop?

How many threads are working in each iteration of the loop?  In the first iteration - only half are.  In the second iteration - only 1/4 are and so on.
It feels wasteful, and it kind of is, but many times reductions are the best way to solve a problem.

#### Set the final value
```
    if (threadIdx.x == 0) {
        // Assume that result is initialized to zero before this kernel using cudaMemset
        atomicAdd(result, sdata[0]);
    }
```

If there are multiple blocks, then each will produce a sum.  If we just added to `result` then they could overwrite each other.  `atomicAdd` ensures 
that all the blocks sum properly.

### Filter Wrapper Function
There is a comment in the above code that states that result must be set to zero before our kernel is called.  This could lead to some really annoying
bugs where someone forgets to initialize the result before calling our kernel.  The solution to this is to write wrapper functions.

Our wrapper can also produce a more universal interface.  If the user needs access to CUDA specific functions and objects, then they have to use the 
NVCC compiler.  If we write our wrapper function without anything CUDA specific (and compile it into a library), then we users can choose to use a
different compiler and still link against our kernel.

Let's write a wrapper for our FIR Filter kernel.

#### Function Signature
```
void sum_function(int* out, const int* in, size_t in_size,
                  const kernel_params& params,
                  CudaStream& stream) {
}
```

#### Data Types
`kernel_params` is a struct that we define that allows us to specify the number of threads and blocks to use.  This allows us to fine tune
kernel performance.

`CudaStream` is an RAII (Resource Acquisition Is Initialization) wrapper around CUDA streams.  This allows us to coordinate which kernels
run at the same time versus blocking in order.

#### Filling in the Function
```
void sum_function(int* out, const int* in, size_t in_size,
                  const kernel_params& params,
                  CudaStream& stream) {

    cudaMemsetAsync(out, 0, sizeof(int), stream.get());
    sum_kernel<<<params.grid_size, params.block_size, 0, stream.get()>>>(out, in, in_size);
}
```

Note that we call `cudaMemsetAsync` on the output data before the kernel executes.  Both kernels execute in the same stream so we know that
the first kernel will finish before the second one starts.

Also note that we don't sync the stream or return the sum here.  We don't know where the result is needed.  If it is needed for another GPU
kernel, then there is no need to copy it back to the host or wait for it to finish.

### Challenges

#### Challenge 1 - Tuner Optimization
Each thread has an initial phase value and then steps the phase by `get_stride()`.  We can rotate a complex number by multiplying
by a complex exponentially with the desired phase. (e.g. `x * exp(j*2*pi*phase)` rotates x by phase.)  We can rotate our phase
using this idea too.

```
shift1 = exp(j*2*pi*init_phase)
offset = exp(j*2*pi*phase_offset)

shift2 = shift1 * offset
shift3 = shift2 * offset
...
```

Update the kernel to calculate the the initial phase and offset outside of the loop using sincosf.  Inside the for loop just use
multiplies to rotate the input vector and update the phase offset. 

#### Challenge 2 - Max Reduction Kernel
Modify the sum kernel wrapper to use the shared memory kernel that we wrote.  Recall that the syntax for launching a kernel is
`<<<blocks, threads, shared_memory_size, stream>>>.  What value do you need to put in for shared memory?

How does the performance compare to the original kernel?

### Challenge 3 - CCCL Reduction
We can just use code from the cccl.  This can be much easier than writing and testing our own kernels.  Frequently these kernels have better performance than the kernels we could write as well.

Look at `cub::DeviceReduce::Sum` and `thrust::reduce`.  How does the performance of these kernels compare to that the kernels that we previously wrote?
