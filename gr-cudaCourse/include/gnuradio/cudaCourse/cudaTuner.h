/* -*- c++ -*- */
/*
 * Copyright 2025 gr-cudaCourse author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDACOURSE_CUDATUNER_H
#define INCLUDED_CUDACOURSE_CUDATUNER_H

#include <gnuradio/cudaCourse/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
  namespace cudaCourse {

    /*!
     * \brief <+description of block+>
     * \ingroup cudaCourse
     *
     */
    class CUDACOURSE_API cudaTuner : virtual public gr::sync_block
    {
     public:
      typedef std::shared_ptr<cudaTuner> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of cudaCourse::cudaTuner.
       *
       * To avoid accidental use of raw pointers, cudaCourse::cudaTuner's
       * constructor is in a private implementation
       * class. cudaCourse::cudaTuner::make is the public interface for
       * creating new instances.
       */
      static sptr make(size_t blockSize, size_t chunkSize, float sampleRate, float freq, float phase);
    };

  } // namespace cudaCourse
} // namespace gr

#endif /* INCLUDED_CUDACOURSE_CUDATUNER_H */
