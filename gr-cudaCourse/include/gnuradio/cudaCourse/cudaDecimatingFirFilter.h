/* -*- c++ -*- */
/*
 * Copyright 2025 gr-cudaCourse author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDACOURSE_CUDADECIMATINGFIRFILTER_H
#define INCLUDED_CUDACOURSE_CUDADECIMATINGFIRFILTER_H

#include <gnuradio/cudaCourse/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace cudaCourse {

    /*!
     * \brief <+description of block+>
     * \ingroup cudaCourse
     *
     */
    class CUDACOURSE_API cudaDecimatingFirFilter : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<cudaDecimatingFirFilter> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of cudaCourse::cudaDecimatingFirFilter.
       *
       * To avoid accidental use of raw pointers, cudaCourse::cudaDecimatingFirFilter's
       * constructor is in a private implementation
       * class. cudaCourse::cudaDecimatingFirFilter::make is the public interface for
       * creating new instances.
       */
      static sptr make(size_t blockSize, size_t chunkSize, size_t decimationFactor, const std::vector<float>& taps);
    };

  } // namespace cudaCourse
} // namespace gr

#endif /* INCLUDED_CUDACOURSE_CUDADECIMATINGFIRFILTER_H */
