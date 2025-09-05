/* -*- c++ -*- */
/*
 * Copyright 2025 gr-cudaCourse author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/io_signature.h>
#include <gnuradio/cudaCourse/cuda_buffer.h>
#include <kernels/cuda/cuda_filter.hpp>
#include "cudaDecimatingFirFilter_impl.h"

namespace gr {
  namespace cudaCourse {

    using input_type = gr_complex;
    using output_type = gr_complex;
    cudaDecimatingFirFilter::sptr
    cudaDecimatingFirFilter::make(size_t blockSize, size_t chunkSize, size_t decimationFactor, const std::vector<float>& taps)
    {
      return gnuradio::make_block_sptr<cudaDecimatingFirFilter_impl>(
        blockSize, chunkSize, decimationFactor, taps);
    }


    /*
     * The private constructor
     */
    cudaDecimatingFirFilter_impl::cudaDecimatingFirFilter_impl(size_t blockSize, size_t chunkSize, size_t decimationFactor, const std::vector<float>& taps)
      : gr::block("cudaDecimatingFirFilter",
              gr::io_signature::make(1 /* min inputs */, 1 /* max inputs */, sizeof(input_type), cuda_buffer::type),
              gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */, sizeof(output_type), cuda_buffer::type)),
	      d_blockSize(blockSize),
	      d_decimationFactor(decimationFactor)
    {
        // Reverse Taps
	// Cuda Memcpy input Taps to Cuda Vector taps
	std::vector<float> temp_taps(taps);
	std::reverse(temp_taps.begin(), temp_taps.end());
        d_taps.resize(temp_taps.size());
	cuda::memcpy(d_taps.data(), temp_taps.data(), (temp_taps.size() * sizeof(float)), d_stream);
	d_stream.sync();
	this->set_output_multiple(chunkSize);
	this->set_history(taps.size());
    }

    /*
     * Our virtual destructor.
     */
    cudaDecimatingFirFilter_impl::~cudaDecimatingFirFilter_impl()
    {
    }

    void
    cudaDecimatingFirFilter_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      /* <+forecast+> e.g. ninput_items_required[0] = noutput_items */
        ninput_items_required[0] = noutput_items * d_decimationFactor + history() - 1;
    }

    int
    cudaDecimatingFirFilter_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      auto in = static_cast<const input_type*>(input_items[0]);
      auto out = static_cast<output_type*>(output_items[0]);

      // Do <+signal processing+>
      auto params = cuda::default_kernel_params(noutput_items, d_blockSize);
      cuda::kernels::decimate_filterN(out, noutput_items,
                                      in, (noutput_items * d_decimationFactor + history() - 1),
				      d_taps.data(), d_taps.size(),
				      d_decimationFactor,
				      params, 
		                      d_stream);
      d_stream.sync();

      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (noutput_items * d_decimationFactor);

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace cudaCourse */
} /* namespace gr */
