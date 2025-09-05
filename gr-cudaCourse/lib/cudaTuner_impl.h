/* -*- c++ -*- */
/*
 * Copyright 2025 gr-cudaCourse author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDACOURSE_CUDATUNER_IMPL_H
#define INCLUDED_CUDACOURSE_CUDATUNER_IMPL_H

#include <gnuradio/cudaCourse/cudaTuner.h>
#include <kernels/cuda/wrapper.hpp>

namespace gr {
  namespace cudaCourse {

    class cudaTuner_impl : public cudaTuner
    {
     private:
      cuda::CudaStream d_stream;
      size_t d_blockSize;
      float  d_sampleRate;
      float  d_freq;
      float  d_phase;

     public:
      cudaTuner_impl(size_t blockSize, size_t chunkSize, float sampleRate, float freq, float phase);
      ~cudaTuner_impl();

      // Where all the action really happens
      int work(
              int noutput_items,
              gr_vector_const_void_star &input_items,
              gr_vector_void_star &output_items
      );
    };

  } // namespace cudaCourse
} // namespace gr

#endif /* INCLUDED_CUDACOURSE_CUDATUNER_IMPL_H */
