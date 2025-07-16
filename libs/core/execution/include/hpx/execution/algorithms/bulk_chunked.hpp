//  Copyright (c) 2024 Laxmi Arvapally
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/executors/parallel_scheduler.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/counting_shape.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        constexpr std::size_t bulk_chunked_default_chunk_size = 16;
        constexpr std::size_t bulk_chunked_max_chunks = 1024;
        constexpr std::size_t bulk_chunked_max_threads = 256;

        template <typename Shape>
        constexpr std::size_t sanitize_shape_size(Shape const& shape) noexcept
        {
            if constexpr (std::is_integral_v<Shape>)
            {
                if (shape <= 0)
                    return 0;
                if (static_cast<std::uint64_t>(shape) >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max()))
                {
                    return std::numeric_limits<std::size_t>::max();
                }
                return static_cast<std::size_t>(shape);
            }
            else
            {
                return static_cast<std::size_t>(hpx::util::size(shape));
            }
        }

        constexpr std::size_t calculate_optimal_chunk_size(
            std::size_t total_size, std::size_t max_threads) noexcept
        {
            if (total_size == 0)
                return 0;

            std::size_t effective_threads =
                (std::min)(max_threads, bulk_chunked_max_threads);
            if (effective_threads == 0)
                effective_threads = 1;

            std::size_t target_chunks = effective_threads * 4;
            std::size_t chunk_size = (total_size + target_chunks - 1) / target_chunks;

            chunk_size = (std::max)(chunk_size, std::size_t(1));
            chunk_size = (std::min)(chunk_size, bulk_chunked_default_chunk_size);

            return chunk_size;
        }

        constexpr std::size_t get_worker_thread_count() noexcept
        {
            auto const* pool = hpx::threads::detail::get_self_or_default_pool();
            return pool ? pool->get_os_thread_count() : 1;
        }

        template <typename F, typename Shape>
        struct chunked_function_wrapper
        {
            F f;
            Shape original_shape;
            std::size_t chunk_size;

            template <typename ChunkIndex>
            void operator()(ChunkIndex chunk_idx) const
            {
                std::size_t start_idx = static_cast<std::size_t>(chunk_idx) * chunk_size;
                std::size_t end_idx = (std::min)(start_idx + chunk_size,
                    sanitize_shape_size(original_shape));

                if constexpr (std::is_integral_v<Shape>)
                {
                    for (std::size_t i = start_idx; i < end_idx; ++i)
                    {
                        f(static_cast<Shape>(i));
                    }
                }
                else
                {
                    auto it = hpx::util::begin(original_shape);
                    std::advance(it, start_idx);
                    for (std::size_t i = start_idx; i < end_idx; ++i, ++it)
                    {
                        f(*it);
                    }
                }
            }
        };

        template <typename Scheduler, typename Sender, typename Shape, typename F>
        constexpr auto apply_chunking(Scheduler&& scheduler, Sender&& sender,
            Shape const& shape, F&& f, std::size_t max_threads = 0)
        {
            std::size_t total_size = sanitize_shape_size(shape);
            if (total_size == 0)
            {
                return hpx::execution::experimental::bulk(
                    HPX_FORWARD(Scheduler, scheduler),
                    HPX_FORWARD(Sender, sender),
                    hpx::util::counting_shape(std::size_t(0)),
                    HPX_FORWARD(F, f));
            }

            std::size_t effective_max_threads = max_threads > 0 ?
                max_threads :
                get_worker_thread_count();

            std::size_t chunk_size =
                calculate_optimal_chunk_size(total_size, effective_max_threads);
            std::size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;

            if (num_chunks > bulk_chunked_max_chunks)
            {
                chunk_size = (total_size + bulk_chunked_max_chunks - 1) /
                    bulk_chunked_max_chunks;
                num_chunks = (total_size + chunk_size - 1) / chunk_size;
            }

            auto chunked_f = chunked_function_wrapper<F, Shape>{
                HPX_FORWARD(F, f), shape, chunk_size};

            return hpx::execution::experimental::bulk(
                HPX_FORWARD(Scheduler, scheduler),
                HPX_FORWARD(Sender, sender),
                hpx::util::counting_shape(num_chunks),
                HPX_MOVE(chunked_f));
        }
    }    // namespace detail

    inline constexpr struct bulk_chunked_t final
      : hpx::functional::detail::tag_fallback<bulk_chunked_t>
    {
    private:
        template <typename Scheduler, typename Sender, typename Shape, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            bulk_chunked_t, Scheduler&& scheduler, Sender&& sender,
            Shape const& shape, F&& f)
        {
            return detail::apply_chunking(HPX_FORWARD(Scheduler, scheduler),
                HPX_FORWARD(Sender, sender), shape, HPX_FORWARD(F, f));
        }

        template <typename Sender, typename Shape, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            bulk_chunked_t, Sender&& sender, Shape const& shape, F&& f)
        {
            auto scheduler = get_parallel_scheduler();
            return detail::apply_chunking(HPX_MOVE(scheduler),
                HPX_FORWARD(Sender, sender), shape, HPX_FORWARD(F, f));
        }

        template <typename Shape, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            bulk_chunked_t, Shape const& shape, F&& f)
        {
            return detail::partial_algorithm<bulk_chunked_t, Shape, F>{
                shape, HPX_FORWARD(F, f)};
        }

        template <typename Scheduler, typename Shape, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            bulk_chunked_t, Scheduler&& scheduler, Shape const& shape, F&& f)
        {
            return detail::partial_algorithm<bulk_chunked_t, Scheduler, Shape, F>{
                HPX_FORWARD(Scheduler, scheduler), shape, HPX_FORWARD(F, f)};
        }
    } bulk_chunked{};

    struct bulk_chunked_with_max_threads_t
    {
        std::size_t max_threads;

        template <typename Scheduler, typename Sender, typename Shape, typename F>
        constexpr auto operator()(Scheduler&& scheduler, Sender&& sender,
            Shape const& shape, F&& f) const
        {
            return detail::apply_chunking(HPX_FORWARD(Scheduler, scheduler),
                HPX_FORWARD(Sender, sender), shape, HPX_FORWARD(F, f),
                max_threads);
        }

        template <typename Sender, typename Shape, typename F>
        constexpr auto operator()(
            Sender&& sender, Shape const& shape, F&& f) const
        {
            auto scheduler = get_parallel_scheduler();
            return detail::apply_chunking(HPX_MOVE(scheduler),
                HPX_FORWARD(Sender, sender), shape, HPX_FORWARD(F, f),
                max_threads);
        }
    };

    constexpr bulk_chunked_with_max_threads_t bulk_chunked_with_max_threads(
        std::size_t max_threads) noexcept
    {
        return {max_threads};
    }

}    // namespace hpx::execution::experimental
