#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 gr-cudaCourse author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
from gnuradio import blocks
from gnuradio import analog

try:
  from gnuradio.cudaCourse import cudaTuner
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.cudaCourse import cudaTuner

class qa_cudaTuner(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()
        self.iv = None
        self.ov = None

    def tearDown(self):
        self.tb = None
        self.iv = None
        self.ov = None

    def test_instance(self):
        instance = cudaTuner(256, 1, 32000, 1000, 0.741) 

    def test_001_descriptive_test_name(self):
        self.samp_rate = samp_rate = 32000
        self.freq = freq = 1000
        self.phase = phase = 0.741
        self.block_size = block_size = 256
        self.chunk_size = chunk_size = 1

        self.blocks_vector_sink_x_0_0 = blocks.vector_sink_c(1, samp_rate)
        self.blocks_vector_sink_x_0 = blocks.vector_sink_c(1, samp_rate)
        self.blocks_head_1 = blocks.head(gr.sizeof_gr_complex*1, samp_rate)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, samp_rate)
        self.analog_sig_source_x_1 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, 1000, 1, 0, 0)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, 2000, 1, 0, .741)
        self.cudaCourse_cudaTuner_0 = cudaTuner(block_size, chunk_size, samp_rate, freq, phase)

        self.tb.connect((self.analog_sig_source_x_1, 0), (self.blocks_head_0, 0))
        self.tb.connect((self.analog_sig_source_x_0, 0), (self.blocks_head_1, 0))
        self.tb.connect((self.blocks_head_0, 0), (self.cudaCourse_cudaTuner_0, 0))
        self.tb.connect((self.blocks_head_1, 0), (self.blocks_vector_sink_x_0, 0))
        self.tb.connect((self.cudaCourse_cudaTuner_0, 0), (self.blocks_vector_sink_x_0_0, 0))

        # set up fg
        self.tb.run()
        # check data
        self.assertFloatTuplesAlmostEqual(self.blocks_vector_sink_x_0_0.data(), self.blocks_vector_sink_x_0.data(), places=3)


if __name__ == '__main__':
    gr_unittest.run(qa_cudaTuner)
