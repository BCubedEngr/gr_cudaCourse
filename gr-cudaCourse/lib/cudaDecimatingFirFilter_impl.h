/* -*- c++ -*- */
/*
 * Copyright 2025 gr-cudaCourse author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDACOURSE_CUDADECIMATINGFIRFILTER_IMPL_H
#define INCLUDED_CUDACOURSE_CUDADECIMATINGFIRFILTER_IMPL_H

#include <gnuradio/cudaCourse/cudaDecimatingFirFilter.h>
#include <kernels/cuda/wrapper.hpp>

namespace gr {
  namespace cudaCourse {

    class cudaDecimatingFirFilter_impl : public cudaDecimatingFirFilter
    {
     private:
      cuda::CudaStream d_stream;
      size_t d_blockSize;
      size_t d_chunkSize;
      size_t d_decimationFactor;
      cuda::vector<float> d_taps;

     public:
      cudaDecimatingFirFilter_impl(size_t blockSize, size_t chunkSize, size_t decimationFactor, const std::vector<float>& taps);
      ~cudaDecimatingFirFilter_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace cudaCourse
} // namespace gr

#endif /* INCLUDED_CUDACOURSE_CUDADECIMATINGFIRFILTER_IMPL_H */
