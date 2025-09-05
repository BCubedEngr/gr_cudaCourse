## Custom Buffers
Before we write our cuda accelerated GNU Radio block we need to understand how data is transferred between blocks in GNU Radio.
Normally, a buffer is allocated for each block output.  These are allocated on the CPU and thus can't be accessed on the GPU.
A simple way to solve this would be to add an extra copy.

Block1 -> CPU Buffer -> GPU Buffer -> GPU Block -> GPU Buffer -> CPU Buffer -> Block 3

This works, but has a lot of issues.  We have to copy our data an extra two times whenever we want to use the GPU.  What if we
have multiple GPU blocks?

CPU Buffer -> GPU Buffer -> GPU Block1 -> GPU Buffer -> CPU Buffer -> GPU Buffer -> GPU Block2 -> GPU Buffer -> CPU Buffer

We need to have a way to have a block write to CPU or GPU buffers and reduce these copies.

https://github.com/gnuradio/gr-cuda

If we look at the code in `lib/cuda_buffer.cc`, there are a few important features.

```
bool cuda_buffer::do_allocate_buffer(size_t final_nitems, size_t sizeof_item) {
...
rc = cudaMallocHost((void**)&d_base, final_nitems * sizeof_item);
...
rc = cudaMalloc((void**)&d_cuda_buf, final_nitems * sizeof_item);
}
```

This allocates two buffers - the host buffer uses dma to allow fast access for CPU and GPU and a unified memory buffer that
contains the memory available in the block's work function.

The `post_work` function is also important.  This allows for custom logic after our block returns from the work function.
https://github.com/gnuradio/gr-cuda/blob/main/lib/cuda_buffer.cc#L148-L201

Notice the special behavior when copying to and from the GPU.  When two GPU blocks are connected together - there is no
additional work.  The moral of the story is that the more GPU blocks we string to together the most efficiently they run.

### Pre-Work Note
CUDA uses a C style API, which leads to some arguably confusing names and allows for some easy to miss errors.  For example,
I can call `cudaMalloc` and forgot to call `cudaFree`.  There are also multiple versions of `cudaMemcpy` and calling the wrong
version can cause significant performance issues.  In the provided code/Docker container for this course, we have 
provided several helper classes to make life easier.

They exist in `cuda/include/kernels/cuda/wrapper.hpp`.  We define custom allocators for the different memory types, a wrapper class
for a cuda stream, and more.

The custom memory allocators allow us to write:

```
cuda::vector<float> x(100);

// Instead of
float* x;
cudaManagedMalloc(&x, 100*sizeof(float));
...
cudaFree(x);
```

## Our first GPU block
We create a GPU block in exactly the same way as a CPU block - using `gr_modtool`.  There are no changes to initial setup.
Inside of the block, there are a handful of changes that we need to make.

Either create a new block `cudaTuner` using gr_modtool or open the block in the provided source code.  The arguments will 
be `size_t blockSize, size_t chunkSize, float sampleRate, float freq, float phase`.


### Use the Cuda Buffer
In the block constructor, we need to change the `io_signature`.

```
gr::sync_block("cudaTuner",
              gr::io_signature::make(1 /* min inputs */, 1 /* max inputs */, sizeof(input_type), cuda_buffer::type),
              gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */, sizeof(output_type), cuda_buffer::type)),
```

Note the extra `cuda_buffer::type` added to the call.  This instructs it that the input and output buffers are going to
use our custom cuda buffer.

### Add a Cuda Stream
Recall that the `default stream` blocks execution of other kernels and streams.  If we have multiple GPU blocks running at
the same time and they are using the same stream or the `default stream` then they will be blocking each other.  We need
for each block to have its own cuda stream.  This will ensure that they can all run at the same time on the GPU.

In the header file, we need to add a private member: `cuda::CudaStream d_stream`.  In the implementation file, we need
to call our kernel and sync to the stream.

```
int cudaTuner_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items) {
    const auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);
    
    float fArg = (d_freq / d_sampleRate) * 2 * M_PI;
    auto params = cuda::default_kernel_params(noutput_items, d_blockSize);
    d_phase = cuda::kernels::vector_tune(out, noutput_items,
		                  in, noutput_items,
	 		 	          fArg,
				          d_phase,
				          params,
        		          d_stream);
    d_stream.sync();
    // Tell runtime system how many output items we produced.
    return noutput_items;
}
```

### Control the ChunkSize and Blocksize
When we run our blocks, we will see that the amount of data that the block processes at a time has a big impact on performance.
When running on the CPU, there is overhead every time we call a function that processes an array of data.  If that array is large
then the amortized cost of the function call is small.  On the GPU, the cost of launching a kernel is much larger.  It involves
sending information to the GPU, the GPU scheduling the kernel, and finally launching it.  If the GPU processes data in too small
of chunks, then it might actually be slower than running on the CPU.

The blocksize controls how many threads are launched with our kernel.  We will also explore the impact this has on our performance.

```
cudaTuner_impl::cudaTuner_impl(size_t blockSize, size_t chunkSize, float sampleRate, float freq, float phase)
      : gr::sync_block("cudaTuner",
              gr::io_signature::make(1 /* min inputs */, 1 /* max inputs */, sizeof(input_type), cuda_buffer::type),
              gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */, sizeof(output_type), cuda_buffer::type)),
              d_blockSize(blockSize),
	      d_sampleRate(sampleRate),
	      d_freq(freq),
	      d_phase(phase)
    {
      set_output_multiple(chunkSize);
    }
```

Using `set_output_multiple(chunkSize)` is an effective way to control how much data the GPU sees at a time.  We store other
values in private variables.

### Testing our block
We use a python QA file to test our block.  In many cases, there is an existing GNU Radio block that should match the functionality
of our CUDA accelerated block.  It is generally best to compare the output of a CPU based block and GPU based block.  **For floating
point math, the values will not match exactly**.  Floating point arithmatic is not associative.  The GPU will do calculations in a 
different order, so values will differ.

Look at 'python/cudaCourse/qa_cudaTuner.py` for the qa tests.


## Decimating FIR Filter Block
We are going to design a decimating FIR Filter block in much the same way.  First we need to add a Decimating FIR Filter kernel

```
__global__ void decimating_fir_filter_kernel(cuda::std::complex<float>* out, const cuda::std::complex<float>* in, size_t size, const float* taps, size_t taps_size, size_t dec) {
    for (size_t i = get_start(); i < size * dec; i += get_stride()) {
        auto out_i = (i / dec);
        auto taps_start = (i % dec);
        float* value = reinterpret_cast<float*>(&out[out_i]);
        for (size_t j = taps_start; j < taps_size; j += dec) {
            atomicAdd(value, in[out_i * dec + j].real() * taps[j] );
            atomicAdd(value + 1, in[out_i * dec + j].imag() * taps[j] );
        }
    }
}
```

Note that this assumes that the filter taps are stored in reverse order.  The wrapper function doesn't handle that because that would be expensive to do on every call.
We could create a wrapper class that handled it for us.

### Wrapper Function
The wrapper function can be found in `gr-cudaCourse/cuda/lib/cuda_filter.cu`.  It is very similiar to the wrapper that we wrote for the max kernel.

### GNU Radio block
We can largely repeat the steps for creating a GNU Radio block as before.  We are not going to walk through all the steps, but will highlight a few key pieces.
The code is available in `gr-cudaCourse/lib/cudaDecimatingFirFilter_impl.cc`.  Note that we created a general block.  It could have been a decimator block, but
we prefer the explicit nature of the general block.

#### History
In the constructor, we call `this->set_history(taps.size());`.  History still works with cuda buffers.  The rest of the class is fairly simple:

- Reverse and copy filter taps
- Create a forecast function
- Call our kernel and sync in the work function

#### Testing our block
The test is almost identical to the tuner block.  It can be found at `gr-cudaCourse/python/cudaCourse/qa_cudaDecimatingFirFilter.py`.  The primary
difference being that we test a variety of decimation values.

## Performance Testing
We have done a lot of work and it is time to reflect on the question - "Was it all worth it?"  We can create a simple flowgraph to see how efficient
the GPU is at the work we are requesting.  We have a sample flowgraph with a probe rate in it at `<insert location here>`.

We can open it up and test our blocks individually as well as in series to see how they perform.

We are going to explore in GRC.

*Note that if we use the "wrong" parameters when calling the blocks, our flowgraph could deadlock and hang.  I believe that this is a bug in the
buffer implementation, but I don't know for sure.  If nothing else, poor parameter selection can lead to significant buffer contention and many
extra copy operations.*

