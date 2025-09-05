#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Decimating FIR Filter QA Design
# GNU Radio version: 3.10.1.1

from gnuradio import analog
from gnuradio import blocks
from gnuradio import cudaCourse
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation




class decFirFiltQa(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Decimating FIR Filter QA Design", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000
        self.variable_low_pass_filter_taps_0 = variable_low_pass_filter_taps_0 = firdes.low_pass(1.0, samp_rate, samp_rate/8,samp_rate/16, window.WIN_HAMMING, 6.76)

        ##################################################
        # Blocks
        ##################################################
        self.fir_filter_xxx_0 = filter.fir_filter_ccf(1, variable_low_pass_filter_taps_0)
        self.fir_filter_xxx_0.declare_sample_delay(0)
        self.cudaCourse_cudaTuner_0 = cudaCourse.cudaTuner(256, 1, 32000, 0, 0.0)
        self.cudaCourse_cudaDecimatingFirFilter_0 = cudaCourse.cudaDecimatingFirFilter(256, 1, 1, variable_low_pass_filter_taps_0)
        self.blocks_vector_sink_x_0_0 = blocks.vector_sink_c(1, 1024)
        self.blocks_vector_sink_x_0 = blocks.vector_sink_c(1, 1024)
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, samp_rate)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, 1000, 1, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.fir_filter_xxx_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.cudaCourse_cudaTuner_0, 0))
        self.connect((self.cudaCourse_cudaDecimatingFirFilter_0, 0), (self.blocks_vector_sink_x_0, 0))
        self.connect((self.cudaCourse_cudaTuner_0, 0), (self.cudaCourse_cudaDecimatingFirFilter_0, 0))
        self.connect((self.fir_filter_xxx_0, 0), (self.blocks_vector_sink_x_0_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_variable_low_pass_filter_taps_0(firdes.low_pass(1.0, self.samp_rate, self.samp_rate/8, self.samp_rate/16, window.WIN_HAMMING, 6.76))
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.blocks_head_0.set_length(self.samp_rate)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)

    def get_variable_low_pass_filter_taps_0(self):
        return self.variable_low_pass_filter_taps_0

    def set_variable_low_pass_filter_taps_0(self, variable_low_pass_filter_taps_0):
        self.variable_low_pass_filter_taps_0 = variable_low_pass_filter_taps_0
        self.fir_filter_xxx_0.set_taps(self.variable_low_pass_filter_taps_0)




def main(top_block_cls=decFirFiltQa, options=None):
    tb = top_block_cls()

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
