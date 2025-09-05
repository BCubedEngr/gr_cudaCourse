#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 gr-cudaCourse author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio.fft import window

try:
  from gnuradio.cudaCourse import cudaDecimatingFirFilter
  from gnuradio.cudaCourse import cudaTuner
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.cudaCourse import cudaDecimatingFirFilter
    from gnuradio.cudaCourse import cudaTuner

class qa_cudaDecimatingFirFilter(gr_unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_instance(self):
        instance = cudaDecimatingFirFilter(256, 1, 2, [1.123, -1.321])

def create_test_function(decimation):
    def test_method(self):
        # Initial Vars
        self.tb = gr.top_block()
        self.chunk_size = chunk_size = 1024
        self.samp_rate = samp_rate = 32 * chunk_size
        self.num_input_items = num_input_items = samp_rate * 10
        self.num_output_items = num_output_items = num_input_items//decimation
        self.variable_low_pass_filter_taps_0 = variable_low_pass_filter_taps_0 = firdes.low_pass(1.0, samp_rate, samp_rate/8,samp_rate/16, window.WIN_HAMMING, 6.76)

        # Setup Blocks
        self.fir_filter_xxx_0 = filter.fir_filter_ccf(decimation, variable_low_pass_filter_taps_0)
        self.fir_filter_xxx_0.declare_sample_delay(0)
        self.cudaCourse_cudaDecimatingFirFilter_0 = cudaDecimatingFirFilter(256, (4*chunk_size)//decimation, decimation, variable_low_pass_filter_taps_0)
        self.cudaCourse_cudaDecimatingFirFilter_0.set_min_output_buffer(64*chunk_size)
        self.blocks_vector_sink_x_0_0 = blocks.vector_sink_c(1, num_output_items)
        self.blocks_vector_sink_x_0 = blocks.vector_sink_c(1, num_output_items)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, num_input_items)
        self.blocks_head_0.set_min_output_buffer(128*chunk_size)
        self.blocks_head_0.set_output_multiple(4*chunk_size)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, 1000, 1, 0, 0)

        # Setup Connections
        self.tb.connect((self.analog_sig_source_x_0, 0), (self.blocks_head_0, 0))
        self.tb.connect((self.analog_sig_source_x_0, 0), (self.fir_filter_xxx_0, 0))
        self.tb.connect((self.blocks_head_0, 0), (self.cudaCourse_cudaDecimatingFirFilter_0, 0))
        self.tb.connect((self.cudaCourse_cudaDecimatingFirFilter_0, 0), (self.blocks_vector_sink_x_0, 0))
        self.tb.connect((self.fir_filter_xxx_0, 0), (self.blocks_vector_sink_x_0_0, 0))

        # set up fg
        self.tb.run()

        # Check data
        if (num_output_items > len(self.blocks_vector_sink_x_0.data())):
            num_output_items = len(self.blocks_vector_sink_x_0.data())
        elif (num_output_items > len(self.blocks_vector_sink_x_0_0.data())):
            num_output_items = len(self.blocks_vector_sink_x_0_0.data())
        self.assertFloatTuplesAlmostEqual(self.blocks_vector_sink_x_0.data()[:num_output_items],
                                          self.blocks_vector_sink_x_0_0.data()[:num_output_items], places=5)
    return test_method
           
test_decimation_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

for i, decimation in enumerate(test_decimation_factors):
    test_name = f'test_case_{i}'
    test_func = create_test_function(decimation)
    setattr(qa_cudaDecimatingFirFilter, test_name, test_func)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(qa_cudaDecimatingFirFilter))
    return suite


if __name__ == '__main__':
    gr_unittest.run(qa_cudaDecimatingFirFilter)
