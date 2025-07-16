//  Copyright (c) 2024 Laxmi Arvapally
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    struct parallel_scheduler
    {
        using execution_category = parallel_execution_tag;

        constexpr parallel_scheduler() noexcept = default;

        explicit parallel_scheduler(
            hpx::threads::thread_pool_base* pool) noexcept
          : underlying_scheduler_(pool)
        {
        }

        friend constexpr bool operator==(
            parallel_scheduler const& lhs,
            parallel_scheduler const& rhs) noexcept
        {
            return lhs.underlying_scheduler_ == rhs.underlying_scheduler_;
        }

        friend constexpr bool operator!=(
            parallel_scheduler const& lhs,
            parallel_scheduler const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        [[nodiscard]] hpx::threads::thread_pool_base* get_thread_pool()
            const noexcept
        {
            return underlying_scheduler_.get_thread_pool();
        }

        template <typename F>
        void execute(F&& f) const
        {
            underlying_scheduler_.execute(HPX_FORWARD(F, f));
        }

        friend constexpr hpx::execution::experimental::
            forward_progress_guarantee
            tag_invoke(
                hpx::execution::experimental::get_forward_progress_guarantee_t,
                parallel_scheduler const&) noexcept
        {
            return hpx::execution::experimental::
                forward_progress_guarantee::parallel;
        }

        friend constexpr auto tag_invoke(
            hpx::execution::experimental::schedule_t,
            parallel_scheduler const& sched)
        {
            return hpx::execution::experimental::schedule(
                sched.underlying_scheduler_);
        }

        template <typename Sender, typename Shape, typename F>
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::bulk_t,
            parallel_scheduler const& sched, Sender&& sender,
            Shape const& shape, F&& f)
        {
            return hpx::execution::experimental::bulk(
                sched.underlying_scheduler_, HPX_FORWARD(Sender, sender),
                shape, HPX_FORWARD(F, f));
        }

        template <typename CPO>
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::get_completion_scheduler_t<CPO>,
            parallel_scheduler const& sched)
        {
            return sched;
        }

        constexpr thread_pool_scheduler const& get_underlying_scheduler()
            const noexcept
        {
            return underlying_scheduler_;
        }

    private:
        thread_pool_scheduler underlying_scheduler_{
            hpx::threads::detail::get_self_or_default_pool()};
    };

    inline parallel_scheduler get_parallel_scheduler() noexcept
    {
        return parallel_scheduler{};
    }

    inline parallel_scheduler get_parallel_scheduler(
        hpx::threads::thread_pool_base* pool) noexcept
    {
        return parallel_scheduler{pool};
    }

}    // namespace hpx::execution::experimental
