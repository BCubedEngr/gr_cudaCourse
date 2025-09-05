#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Cuda Filter Test
# GNU Radio version: 3.10.1.1

from gnuradio import analog
from gnuradio import blocks
from gnuradio import cudaCourse
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation


def snipfcn_snippet_0(self):
    self.blocks_head_0.set_output_multiple((4*self.chunk_size)//self.decimation)


def snippets_main_after_init(tb):
    snipfcn_snippet_0(tb)


class cudaFilterTest(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Cuda Filter Test", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.chunk_size = chunk_size = 1024
        self.samp_rate = samp_rate = 32 * chunk_size
        self.num_input_items = num_input_items = samp_rate * 1000
        self.decimation = decimation = 2
        self.variable_low_pass_filter_taps_0 = variable_low_pass_filter_taps_0 = firdes.low_pass(1.0, samp_rate, samp_rate/8,samp_rate/16, window.WIN_HAMMING, 6.76)
        self.num_output_items = num_output_items = num_input_items//decimation

        ##################################################
        # Blocks
        ##################################################
        self.cudaCourse_cudaDecimatingFirFilter_0 = cudaCourse.cudaDecimatingFirFilter(256, (4 * chunk_size)//decimation, decimation, variable_low_pass_filter_taps_0)
        self.cudaCourse_cudaDecimatingFirFilter_0.set_min_output_buffer(65536)
        self.blocks_probe_rate_0 = blocks.probe_rate(gr.sizeof_gr_complex*1, 500.0, 0.15)
        self.blocks_probe_rate_0.set_min_output_buffer(131072)
        self.blocks_message_debug_0 = blocks.message_debug(True)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, num_input_items)
        self.blocks_head_0.set_min_output_buffer(131072)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, 1000, 1, 0, 0)
        self.analog_sig_source_x_0.set_min_output_buffer(131072)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_probe_rate_0, 'rate'), (self.blocks_message_debug_0, 'print'))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_head_0, 0), (self.cudaCourse_cudaDecimatingFirFilter_0, 0))
        self.connect((self.cudaCourse_cudaDecimatingFirFilter_0, 0), (self.blocks_probe_rate_0, 0))


    def get_chunk_size(self):
        return self.chunk_size

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size
        self.set_samp_rate(32 * self.chunk_size)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_num_input_items(self.samp_rate * 1000)
        self.set_variable_low_pass_filter_taps_0(firdes.low_pass(1.0, self.samp_rate, self.samp_rate/8, self.samp_rate/16, window.WIN_HAMMING, 6.76))
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)

    def get_num_input_items(self):
        return self.num_input_items

    def set_num_input_items(self, num_input_items):
        self.num_input_items = num_input_items
        self.set_num_output_items(self.num_input_items//self.decimation)
        self.blocks_head_0.set_length(self.num_input_items)

    def get_decimation(self):
        return self.decimation

    def set_decimation(self, decimation):
        self.decimation = decimation
        self.set_num_output_items(self.num_input_items//self.decimation)

    def get_variable_low_pass_filter_taps_0(self):
        return self.variable_low_pass_filter_taps_0

    def set_variable_low_pass_filter_taps_0(self, variable_low_pass_filter_taps_0):
        self.variable_low_pass_filter_taps_0 = variable_low_pass_filter_taps_0

    def get_num_output_items(self):
        return self.num_output_items

    def set_num_output_items(self, num_output_items):
        self.num_output_items = num_output_items




def main(top_block_cls=cudaFilterTest, options=None):
    tb = top_block_cls()
    snippets_main_after_init(tb)
    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
