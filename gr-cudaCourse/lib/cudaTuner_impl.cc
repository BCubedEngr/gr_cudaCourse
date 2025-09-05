/* -*- c++ -*- */
/*
 * Copyright 2025 gr-cudaCourse author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/io_signature.h>
#include <gnuradio/cudaCourse/cuda_buffer.h>
#include <kernels/cuda/cuda_tune.hpp>
#include "cudaTuner_impl.h"

namespace gr {
  namespace cudaCourse {

    using input_type = gr_complex;
    using output_type = gr_complex;
    cudaTuner::sptr
    cudaTuner::make(size_t blockSize, size_t chunkSize, float sampleRate, float freq, float phase)
    {
      return gnuradio::make_block_sptr<cudaTuner_impl>(blockSize, chunkSize, sampleRate, freq, phase);
    }


    /*
     * The private constructor
     */
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

    /*
     * Our virtual destructor.
     */
    cudaTuner_impl::~cudaTuner_impl()
    {}

    int
    cudaTuner_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
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

  } /* namespace cudaCourse */
} /* namespace gr */
