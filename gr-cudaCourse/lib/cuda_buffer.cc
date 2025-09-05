/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/block.h>
#include <gnuradio/cudaCourse/cuda_buffer.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace gr {

buffer_type cuda_buffer::type(buftype<cuda_buffer, cuda_buffer>{});

void* cuda_buffer::cuda_memcpy(void* dest, const void* src, std::size_t count) {
    cuda::memcpy(dest, src, count, d_stream, cuda::cudaMemcpyDeviceToDevice);
    d_stream.sync();
    return dest;
}

void* cuda_buffer::cuda_memmove(void* dest, const void* src, std::size_t count) {
    if (count == 0)
        return dest;
    auto dist = std::distance(reinterpret_cast<const char*>(src), reinterpret_cast<const char*>(dest));
    // Round up to see how many copies it will take.  If it is only a few, then we can
    // just do memcpys without needing to allocate an extra buffer
    size_t num_copies = (std::abs(dist) + count - 1) / std::abs(dist);
    // 4 was found empirically to be a good value
    if (num_copies < 4) {
        for (size_t i = 0; i < num_copies; i++) {
            size_t offset;
            // Copy backwards if dest > src
            if (dist > 0)
                offset = (num_copies - i - 1) * dist;
            else
                offset = -i * dist;
            cuda::memcpy(reinterpret_cast<char*>(dest) + offset,
                             reinterpret_cast<const char*>(src) + offset,
                             std::min<ssize_t>(std::abs(dist), count - offset),
                             d_stream,
                             cuda::cudaMemcpyDeviceToDevice);
        }
        d_stream.sync();
    } else {
        // Allocate temp buffer
        _tempBuffer.resize(count);
        cuda::memcpy(_tempBuffer.data(), src, count, d_stream, cuda::cudaMemcpyDeviceToDevice);
        cuda::memcpy(dest, _tempBuffer.data(), count, d_stream, cuda::cudaMemcpyDeviceToDevice);
        d_stream.sync();
    }
    return dest;
}

cuda_buffer::cuda_buffer(int nitems,
                         size_t sizeof_item,
                         uint64_t downstream_lcm_nitems,
                         uint32_t downstream_max_out_mult,
                         block_sptr link,
                         block_sptr buf_owner)
    : buffer_single_mapped(nitems, sizeof_item, downstream_lcm_nitems, downstream_max_out_mult, link, buf_owner) {
    gr::configure_default_loggers(d_logger, d_debug_logger, "cuda_buffer");
    if (!allocate_buffer(nitems))
        throw std::bad_alloc();

    f_cuda_memcpy = [this](void* dest, const void* src, std::size_t count) {
        return this->cuda_memcpy(dest, src, count);
    };
    f_cuda_memmove = [this](void* dest, const void* src, std::size_t count) {
        return this->cuda_memmove(dest, src, count);
    };
}

cuda_buffer::~cuda_buffer() {
    // Free host buffer
    if (d_base != nullptr) {
        cuda::detail::_dma_deallocate(d_base);
        d_base = nullptr;
    }
}

void cuda_buffer::post_work(int nitems) {
#ifdef BUFFER_DEBUG
    std::ostringstream msg;
    msg << "[" << this << "] "
        << "cuda_buffer [" << d_transfer_type << "] -- post_work: " << nitems;
    GR_LOG_DEBUG(d_logger, msg.str());
#endif

    if (nitems <= 0) {
        return;
    }

    // NOTE: when this function is called the write pointer has not yet been
    // advanced so it can be used directly as the source ptr
    switch (d_transfer_type) {
        case transfer_type::HOST_TO_DEVICE: {
            // Copy data from host buffer to device buffer
            void* dest_ptr = &d_cuda_buf[d_write_index * d_sizeof_item];
            cuda::memcpy(dest_ptr,
                             write_pointer(),
                             nitems * d_sizeof_item,
                             d_stream,
                             cuda::cudaMemcpyHostToDevice);
            d_stream.sync();
        } break;

        case transfer_type::DEVICE_TO_HOST: {
            // Copy data from device buffer to host buffer
            void* dest_ptr = &d_base[d_write_index * d_sizeof_item];
            cuda::memcpy(dest_ptr,
                             write_pointer(),
                             nitems * d_sizeof_item,
                             d_stream,
                             cuda::cudaMemcpyDeviceToHost);
            d_stream.sync();
        } break;

        case transfer_type::DEVICE_TO_DEVICE:
            // No op FTW!
            break;

        default:
            std::ostringstream msg2;
            msg2 << "Unexpected context for cuda_buffer: " << d_transfer_type;
            GR_LOG_ERROR(d_logger, msg2.str());
            throw std::runtime_error(msg2.str());
    }

    return;
}

bool cuda_buffer::do_allocate_buffer(size_t final_nitems, size_t sizeof_item) {
#ifdef BUFFER_DEBUG
    {
        std::ostringstream msg;
        msg << "[" << this << "] "
            << "cuda_buffer constructor -- nitems: " << final_nitems;
        GR_LOG_DEBUG(d_logger, msg.str());
    }
#endif

    // This is the pinned host buffer
    // Can a CUDA buffer even use std::unique_ptr ?
    //    d_buffer.reset(new char[final_nitems * sizeof_item]);
    d_base = reinterpret_cast<char*>(cuda::detail::_dma_allocate(final_nitems * sizeof_item));

    // This is the CUDA device buffer
    d_cuda_buf.resize(final_nitems * sizeof_item);

    return true;
}

void* cuda_buffer::write_pointer() {
    void* ptr = nullptr;
    switch (d_transfer_type) {
        case transfer_type::HOST_TO_DEVICE:
            // Write into host buffer
            ptr = &d_base[d_write_index * d_sizeof_item];
            break;

        case transfer_type::DEVICE_TO_HOST:
        case transfer_type::DEVICE_TO_DEVICE:
            // Write into CUDA device buffer
            ptr = &d_cuda_buf[d_write_index * d_sizeof_item];
            break;

        default:
            std::ostringstream msg;
            msg << "Unexpected context for cuda_buffer: " << d_transfer_type;
            GR_LOG_ERROR(d_logger, msg.str());
            throw std::runtime_error(msg.str());
    }

    return ptr;
}

const void* cuda_buffer::_read_pointer(unsigned int read_index) {
    void* ptr = nullptr;
    switch (d_transfer_type) {
        case transfer_type::HOST_TO_DEVICE:
        case transfer_type::DEVICE_TO_DEVICE:
            // Read from "device" buffer
            ptr = &d_cuda_buf[read_index * d_sizeof_item];
            break;

        case transfer_type::DEVICE_TO_HOST:
            // Read from host buffer
            ptr = &d_base[read_index * d_sizeof_item];
            break;

        default:
            std::ostringstream msg;
            msg << "Unexpected context for cuda_buffer: " << d_transfer_type;
            GR_LOG_ERROR(d_logger, msg.str());
            throw std::runtime_error(msg.str());
    }

    return ptr;
}

bool cuda_buffer::input_blocked_callback(int items_required, int items_avail, unsigned read_index) {
#ifdef BUFFER_DEBUG
    std::ostringstream msg;
    msg << "[" << this << "] "
        << "cuda_buffer [" << d_transfer_type << "] -- input_blocked_callback";
    GR_LOG_DEBUG(d_logger, msg.str());
#endif

    bool rc = false;
    switch (d_transfer_type) {
        case transfer_type::HOST_TO_DEVICE:
        case transfer_type::DEVICE_TO_DEVICE:
            // Adjust "device" buffer
            rc = input_blocked_callback_logic(items_required,
                                              items_avail,
                                              read_index,
                                              d_cuda_buf.data(),
                                              f_cuda_memcpy,
                                              f_cuda_memmove);
            break;

        case transfer_type::DEVICE_TO_HOST:
            // Adjust host buffer
            // This logic was wrong for a blocked output buffer, but I'm not sure if it is correct
            // here. This needs to be verified at some point.
            rc =
              input_blocked_callback_logic(items_required, items_avail, read_index, d_base, std::memcpy, std::memmove);
            break;

        default:
            std::ostringstream msg2;
            msg2 << "Unexpected context for cuda_buffer: " << d_transfer_type;
            GR_LOG_ERROR(d_logger, msg2.str());
            throw std::runtime_error(msg2.str());
    }

    return rc;
}

bool cuda_buffer::output_blocked_callback(int output_multiple, bool force) {
#ifdef BUFFER_DEBUG
    std::ostringstream msg;
    msg << "[" << this << "] "
        << "host_buffer [" << d_transfer_type << "] -- output_blocked_callback";
    GR_LOG_DEBUG(d_logger, msg.str());
#endif

    bool rc = false;
    switch (d_transfer_type) {
        case transfer_type::HOST_TO_DEVICE:
        case transfer_type::DEVICE_TO_HOST:
        case transfer_type::DEVICE_TO_DEVICE:
            // Adjust "device" buffer
            rc = output_blocked_callback_logic(output_multiple, force, d_cuda_buf.data(), f_cuda_memmove);
            break;

        default:
            std::ostringstream msg2;
            msg2 << "Unexpected context for cuda_buffer: " << d_transfer_type;
            GR_LOG_ERROR(d_logger, msg2.str());
            throw std::runtime_error(msg2.str());
    }

    return rc;
}

buffer_sptr cuda_buffer::make_buffer(int nitems,
                                     size_t sizeof_item,
                                     uint64_t downstream_lcm_nitems,
                                     uint32_t downstream_max_out_mult,
                                     block_sptr link,
                                     block_sptr buf_owner) {
    return buffer_sptr(
      new cuda_buffer(nitems, sizeof_item, downstream_lcm_nitems, downstream_max_out_mult, link, buf_owner));
}

}  // namespace gr
