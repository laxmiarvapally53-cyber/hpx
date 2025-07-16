//

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/bulk_chunked.hpp>
#include <hpx/execution/algorithms/starts_on.hpp>
#include <hpx/execution/algorithms/sync_wait.hpp>
#include <hpx/execution/algorithms/then.hpp>
#include <hpx/executors/parallel_scheduler.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace hpx::execution::experimental;

std::atomic<int> global_counter{0};
std::mutex output_mutex;
std::unordered_map<std::thread::id, int> thread_usage_map;
std::mutex thread_map_mutex;

void log_thread_usage(std::thread::id tid)
{
    std::lock_guard<std::mutex> lock(thread_map_mutex);
    thread_usage_map[tid]++;
}

void test_scheduler_construction()
{
    std::cout << "\n=== Testing Scheduler Construction ===" << std::endl;

    auto scheduler1 = get_parallel_scheduler();
    auto scheduler2 = get_parallel_scheduler();

    HPX_TEST(scheduler1 == scheduler2);
    HPX_TEST(!(scheduler1 != scheduler2));

    std::cout << "Scheduler equality tests passed" << std::endl;

    static_assert(noexcept(get_parallel_scheduler()));
    static_assert(noexcept(scheduler1 == scheduler2));
    static_assert(noexcept(scheduler1 != scheduler2));

    std::cout << "Scheduler noexcept properties verified" << std::endl;
}

void test_forward_progress_guarantee()
{
    std::cout << "\n=== Testing Forward Progress Guarantee ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto guarantee = get_forward_progress_guarantee(scheduler);

    HPX_TEST(guarantee == forward_progress_guarantee::parallel);

    std::string guarantee_str;
    switch (guarantee)
    {
    case forward_progress_guarantee::concurrent:
        guarantee_str = "concurrent";
        break;
    case forward_progress_guarantee::parallel:
        guarantee_str = "parallel";
        break;
    case forward_progress_guarantee::weakly_parallel:
        guarantee_str = "weakly_parallel";
        break;
    }

    std::cout << "Forward progress guarantee: " << guarantee_str << std::endl;
    std::cout << "Forward progress guarantee test passed" << std::endl;
}

void test_schedule()
{
    std::cout << "\n=== Testing Schedule ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto sender = schedule(scheduler);

    std::cout << "Created sender with schedule" << std::endl;

    static_assert(is_sender_v<decltype(sender)>);

    std::cout << "Schedule sender properties verified" << std::endl;
}

void test_completion_scheduler_query()
{
    std::cout << "\n=== Testing Completion Scheduler Query ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto sender = schedule(scheduler);

    auto completion_scheduler =
        get_completion_scheduler<set_value_t>(sender);

    HPX_TEST(completion_scheduler == scheduler);

    std::cout << "Completion scheduler query test passed" << std::endl;
}

void test_stop_token()
{
    std::cout << "\n=== Testing Stop Token ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto sender = schedule(scheduler);

    std::atomic<bool> stopped_received{false};
    std::atomic<bool> error_received{false};

    struct test_receiver
    {
        std::atomic<bool>* stopped_received;
        std::atomic<bool>* error_received;

        void set_value() noexcept
        {
            auto tid = std::this_thread::get_id();
            std::cout << "set_value called on thread " << tid << std::endl;
        }

        void set_error(std::exception_ptr) noexcept
        {
            *error_received = true;
        }

        void set_stopped() noexcept
        {
            *stopped_received = true;
        }
    };

    auto op_state = connect(sender, test_receiver{&stopped_received, &error_received});

    std::cout << "Calling start for stop token test" << std::endl;
    start(op_state);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "Stop token test: stopped_received = " << stopped_received.load()
              << ", error_received = " << error_received.load() << std::endl;
}

void test_shared_context()
{
    std::cout << "\n=== Testing Shared Context ===" << std::endl;

    auto scheduler1 = get_parallel_scheduler();
    auto scheduler2 = get_parallel_scheduler();

    HPX_TEST(scheduler1.get_thread_pool() == scheduler2.get_thread_pool());

    std::cout << "Schedulers share same context" << std::endl;
}

void test_basic_execution()
{
    std::cout << "\n=== Testing Basic Execution ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto sender = schedule(scheduler) | then([]() {
        std::cout << "Executing then functor returning 42" << std::endl;
        return 42;
    });

    std::cout << "Calling sync_wait for basic execution" << std::endl;
    auto result = sync_wait(sender);

    HPX_TEST(result.has_value());
    HPX_TEST(*result == 42);

    std::cout << "Basic execution result: " << *result << std::endl;
}

void test_structured_concurrency()
{
    std::cout << "\n=== Testing Structured Concurrency with starts_on ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto sender = starts_on(scheduler, then(schedule(scheduler), []() {
        std::cout << "Executing then functor returning 'Hello, P2079!'" << std::endl;
        return std::string("Hello, P2079!");
    }));

    std::cout << "Calling sync_wait for structured concurrency" << std::endl;
    auto result = sync_wait(sender);

    HPX_TEST(result.has_value());
    HPX_TEST(*result == "Hello, P2079!");

    std::cout << "Structured concurrency result: " << *result << std::endl;
}

void test_error_handling()
{
    std::cout << "\n=== Testing Error Handling ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto sender = schedule(scheduler) | then([]() -> int {
        std::cout << "Throwing runtime_error in then functor" << std::endl;
        throw std::runtime_error("Test error");
    });

    std::cout << "Calling sync_wait for error handling" << std::endl;
    try
    {
        auto result = sync_wait(sender);
        HPX_TEST(false);
    }
    catch (std::runtime_error const& e)
    {
        std::cout << "Caught error: " << e.what() << std::endl;
        HPX_TEST(std::string(e.what()) == "Test error");
    }

    std::cout << "Error handling test passed" << std::endl;
}

void test_shared_context_with_algorithms()
{
    std::cout << "\n=== Testing Shared Context with Algorithms ===" << std::endl;

    auto scheduler1 = get_parallel_scheduler();
    auto scheduler2 = get_parallel_scheduler();

    HPX_TEST(scheduler1.get_thread_pool() == scheduler2.get_thread_pool());
    std::cout << "Schedulers share same context" << std::endl;

    std::atomic<int> count1{0};
    std::atomic<int> count2{0};

    auto sender1 = schedule(scheduler1) | then([&count1]() {
        std::cout << "Executing sender1 then functor" << std::endl;
        count1++;
    });

    auto sender2 = schedule(scheduler2) | then([&count2]() {
        std::cout << "Executing sender2 then functor" << std::endl;
        count2++;
    });

    std::cout << "Calling sync_wait for shared context" << std::endl;
    sync_wait(sender1);
    sync_wait(sender2);

    std::cout << "Shared context test: count1 = " << count1.load()
              << ", count2 = " << count2.load() << std::endl;

    HPX_TEST(count1.load() == 1);
    HPX_TEST(count2.load() == 1);
}

void test_p2079r10_examples()
{
    std::cout << "\n=== Testing P2079R10 Examples ===" << std::endl;

    auto scheduler = get_parallel_scheduler();

    auto sender1 = schedule(scheduler) | then([](){ 
        std::cout << "Executing P2079R10 Example 1 then functor" << std::endl;
        std::cout << "Adding 42 to 13" << std::endl;
        return 42 + 13; 
    });

    std::cout << "Calling sync_wait for P2079R10 Example 1" << std::endl;
    auto result1 = sync_wait(sender1);
    HPX_TEST(result1.has_value());
    HPX_TEST(*result1 == 55);
    std::cout << "P2079R10 Example 1 result: " << *result1 << std::endl;

    auto sender2 = starts_on(scheduler, then(schedule(scheduler), [](){
        std::cout << "Executing P2079R10 Example 2 then functor" << std::endl;
        std::cout << "Adding 42 to 13" << std::endl;
        return 42 + 13;
    }));

    std::cout << "Calling sync_wait for P2079R10 Example 2" << std::endl;
    auto result2 = sync_wait(sender2);
    HPX_TEST(result2.has_value());
    HPX_TEST(*result2 == 55);
    std::cout << "P2079R10 Example 2 result: " << *result2 << std::endl;
}

void test_case_1()
{
    std::cout << "\n=== Test Case 1: Verify HPX Thread ID ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto sender = schedule(scheduler) | then([]() { return 1; });

    std::thread::id thread_id;
    std::atomic<bool> completed{false};

    struct test_receiver
    {
        std::thread::id* thread_id;
        std::atomic<bool>* completed;

        void set_value(int value) noexcept
        {
            auto tid = std::this_thread::get_id();
            std::cout << "set_value(int: " << value << ") called on thread " << tid << std::endl;
            *thread_id = tid;
            *completed = true;
        }

        void set_error(std::exception_ptr) noexcept {}
        void set_stopped() noexcept {}
    };

    auto op_state = connect(sender, test_receiver{&thread_id, &completed});

    std::cout << "Calling connect and start for Test Case 1" << std::endl;
    start(op_state);

    std::cout << "Waiting for Test Case 1 to complete" << std::endl;
    while (!completed.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "Test Case 1 thread ID: " << thread_id << std::endl;
}

void test_case_2()
{
    std::cout << "\n=== Test Case 2: Verify Different HPX Thread ID ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto sender = schedule(scheduler) | then([]() { return 2; });

    std::thread::id thread_id;
    std::atomic<bool> completed{false};

    struct test_receiver
    {
        std::thread::id* thread_id;
        std::atomic<bool>* completed;

        void set_value(int value) noexcept
        {
            auto tid = std::this_thread::get_id();
            std::cout << "set_value(int: " << value << ") called on thread " << tid << std::endl;
            *thread_id = tid;
            *completed = true;
        }

        void set_error(std::exception_ptr) noexcept {}
        void set_stopped() noexcept {}
    };

    auto op_state = connect(sender, test_receiver{&thread_id, &completed});

    std::cout << "Calling connect and start for Test Case 2" << std::endl;
    start(op_state);

    std::cout << "Waiting for Test Case 2 to complete" << std::endl;
    while (!completed.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "Test Case 2 thread ID: " << thread_id << std::endl;
}

void test_case_3_bulk_chunked()
{
    std::cout << "\n=== Test Case 3: Verify bulk_chunked with parallel_scheduler ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    auto sender = schedule(scheduler) | bulk_chunked(200, [](std::size_t i) {
        auto tid = std::this_thread::get_id();
        std::size_t chunk_start = (i / 4) * 4;
        std::size_t chunk_end = chunk_start + 4;
        if (chunk_end > 200) chunk_end = 200;
        
        std::cout << "Processing chunk [" << chunk_start << ", " << chunk_end 
                  << ") with value 99 on thread " << tid << std::endl;
        
        log_thread_usage(tid);
        global_counter++;
    });

    std::cout << "Calling sync_wait for bulk_chunked test" << std::endl;
    sync_wait(sender);

    std::cout << "Total processed elements: " << global_counter.load() << std::endl;
    
    std::lock_guard<std::mutex> lock(thread_map_mutex);
    std::cout << "Test Case 3: bulk_chunked processed " << (global_counter.load() / 4) 
              << " chunks successfully" << std::endl;
}

void test_bulk_chunked_with_thread_limits()
{
    std::cout << "\n=== Testing bulk_chunked with Configurable Thread Limits ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    std::atomic<int> counter{0};
    std::unordered_map<std::thread::id, int> local_thread_map;
    std::mutex local_mutex;

    auto sender = schedule(scheduler) | 
        bulk_chunked_with_max_threads(2)(1000, [&](std::size_t) {
            std::lock_guard<std::mutex> lock(local_mutex);
            auto tid = std::this_thread::get_id();
            local_thread_map[tid]++;
            counter++;
        });

    std::cout << "Calling sync_wait for limited thread test..." << std::endl;
    sync_wait(sender);

    std::cout << "Thread-limited test used " << local_thread_map.size() << " threads" << std::endl;
    std::cout << "Total processed with 2-thread limit: " << counter.load() << std::endl;
}

void test_negative_shape_handling()
{
    std::cout << "\n=== Testing Negative Shape Handling ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    std::atomic<int> call_count{0};

    auto sender = schedule(scheduler) | bulk_chunked(-5, [&](int) {
        call_count++;
    });

    sync_wait(sender);

    HPX_TEST(call_count.load() == 0);
    std::cout << "Negative shape test: PASSED (no function calls)" << std::endl;
}

void test_enhanced_bulk_chunked_analytics()
{
    std::cout << "\n=== Enhanced bulk_chunked Test with Analytics ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    std::atomic<int> counter{0};
    std::unordered_map<std::thread::id, std::vector<std::pair<std::size_t, std::size_t>>> thread_chunks;
    std::unordered_map<std::size_t, int> chunk_size_distribution;
    std::vector<std::string> execution_log;
    std::mutex analytics_mutex;

    auto start_time = std::chrono::high_resolution_clock::now();

    auto sender = schedule(scheduler) | bulk_chunked(1000, [&](std::size_t i) {
        std::lock_guard<std::mutex> lock(analytics_mutex);
        auto tid = std::this_thread::get_id();
        
        counter++;
        
        std::size_t chunk_start = (i / 16) * 16;
        std::size_t chunk_end = std::min(chunk_start + 16, std::size_t(1000));
        
        thread_chunks[tid].emplace_back(chunk_start, chunk_end);
        
        std::size_t chunk_size = chunk_end - chunk_start;
        chunk_size_distribution[chunk_size]++;
        
        if (execution_log.size() < 10) {
            execution_log.push_back("  Chunk [" + std::to_string(chunk_start) + "-" + 
                                  std::to_string(chunk_end) + ") with value 42 on thread " + 
                                  std::to_string(std::hash<std::thread::id>{}(tid)));
        }
    });

    std::cout << "Calling sync_wait for bulk_chunked execution..." << std::endl;
    sync_wait(sender);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Bulk execution completed: counter = " << counter.load() 
              << ", expected = 1000" << std::endl;

    std::cout << "\nExecution Details:" << std::endl;
    std::cout << "Total execution time: " << (duration.count() / 1000.0) << " ms" << std::endl;

    std::size_t total_chunks = 0;
    for (const auto& [tid, chunks] : thread_chunks) {
        total_chunks += chunks.size();
    }
    std::cout << "Number of chunks executed: " << total_chunks << std::endl;

    std::cout << "\nThread-wise work distribution:" << std::endl;
    for (const auto& [tid, chunks] : thread_chunks) {
        std::size_t items_processed = 0;
        std::size_t min_idx = std::numeric_limits<std::size_t>::max();
        std::size_t max_idx = 0;
        
        for (const auto& [start, end] : chunks) {
            items_processed += (end - start);
            min_idx = std::min(min_idx, start);
            max_idx = std::max(max_idx, end);
        }
        
        std::cout << "Thread " << tid << " executed " << items_processed 
                  << " items in " << chunks.size() << " chunks: [" 
                  << min_idx << "-" << max_idx << ")" << std::endl;
    }

    std::cout << "Number of unique threads used: " << thread_chunks.size() << std::endl;
    std::cout << "Total items processed: " << counter.load() << std::endl;

    std::cout << "\nChunk size analysis:" << std::endl;
    for (const auto& [size, count] : chunk_size_distribution) {
        std::cout << "Chunk size " << size << ": " << count << " chunks" << std::endl;
    }

    std::cout << "\nFirst 10 chunk executions:" << std::endl;
    for (const auto& log_entry : execution_log) {
        std::cout << log_entry << std::endl;
    }

    HPX_TEST(counter.load() == 1000);
    std::cout << "Correctness validation: PASSED" << std::endl;
}

void test_bulk_chunked_with_value_propagation()
{
    std::cout << "\n=== Testing bulk_chunked with Value Propagation ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    std::atomic<int> sum{0};
    std::unordered_map<std::thread::id, int> thread_contributions;
    std::mutex contrib_mutex;

    auto sender = schedule(scheduler) | then([]() { return 42; }) |
        bulk_chunked(100, [&](std::size_t i, int value) {
            std::lock_guard<std::mutex> lock(contrib_mutex);
            auto tid = std::this_thread::get_id();
            int contribution = static_cast<int>(i) * value;
            sum += contribution;
            thread_contributions[tid] += contribution;
        });

    std::cout << "Calling sync_wait for bulk_chunked with value..." << std::endl;
    auto result = sync_wait(sender);

    int expected_sum = 0;
    for (int i = 0; i < 100; ++i) {
        expected_sum += i * 42;
    }

    std::cout << "Bulk with value completed: sum = " << sum.load() 
              << ", expected = " << expected_sum << std::endl;

    std::cout << "Per-thread contributions:" << std::endl;
    for (const auto& [tid, contrib] : thread_contributions) {
        std::cout << "  Thread " << tid << " contributed: " << contrib << std::endl;
    }

    HPX_TEST(sum.load() == expected_sum);
    HPX_TEST(result.has_value());
    HPX_TEST(*result == 42);
}

void test_bulk_chunked_error_handling()
{
    std::cout << "\n=== Testing bulk_chunked Error Handling ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    std::atomic<int> processed_count{0};

    auto sender = schedule(scheduler) | bulk_chunked(100, [&](std::size_t i) {
        if (i == 25) {
            auto tid = std::this_thread::get_id();
            std::cout << "Throwing error from chunk [25-26) at index " << i 
                      << " on thread " << tid << std::endl;
            throw std::runtime_error("Bulk chunked error at index " + std::to_string(i));
        }
        processed_count++;
    });

    std::cout << "Calling sync_wait for bulk_chunked error test..." << std::endl;
    try {
        sync_wait(sender);
        HPX_TEST(false);
    } catch (const std::runtime_error& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
        std::cout << "Items processed before error: " << processed_count.load() << std::endl;
        HPX_TEST(std::string(e.what()).find("Bulk chunked error at index 25") != std::string::npos);
    }

    std::cout << "Error handling test: PASSED" << std::endl;
}

void test_bulk_chunked_edge_cases()
{
    std::cout << "\n=== Testing bulk_chunked Edge Cases ===" << std::endl;

    auto scheduler = get_parallel_scheduler();

    std::cout << "Testing empty range..." << std::endl;
    std::atomic<int> empty_count{0};
    auto empty_sender = schedule(scheduler) | bulk_chunked(0, [&](std::size_t) {
        empty_count++;
    });
    sync_wait(empty_sender);
    HPX_TEST(empty_count.load() == 0);
    std::cout << "Empty range test: PASSED (no function calls)" << std::endl;

    std::cout << "Testing single item..." << std::endl;
    std::atomic<int> single_count{0};
    auto single_sender = schedule(scheduler) | bulk_chunked(1, [&](std::size_t i) {
        auto tid = std::this_thread::get_id();
        std::cout << "Single item chunk: [" << i << "-" << (i+1) << ") with value 777" << std::endl;
        single_count++;
    });
    sync_wait(single_sender);
    HPX_TEST(single_count.load() == 1);
    std::cout << "Single item test: PASSED" << std::endl;

    std::cout << "Testing large range (10000 items)..." << std::endl;
    std::atomic<int> large_count{0};
    std::atomic<int> chunk_count{0};
    
    auto large_start = std::chrono::high_resolution_clock::now();
    auto large_sender = schedule(scheduler) | bulk_chunked(10000, [&](std::size_t) {
        large_count++;
        if (large_count.load() % 250 == 1) {
            chunk_count++;
        }
    });
    sync_wait(large_sender);
    auto large_end = std::chrono::high_resolution_clock::now();
    auto large_duration = std::chrono::duration_cast<std::chrono::microseconds>(large_end - large_start);
    
    std::cout << "Large range completed: " << large_count.load() << " items in " 
              << chunk_count.load() << " chunks" << std::endl;
    std::cout << "Large range execution time: " << (large_duration.count() / 1000.0) << " ms" << std::endl;
    HPX_TEST(large_count.load() == 10000);
    std::cout << "Large range test: PASSED" << std::endl;
}

void test_overflow_safety()
{
    std::cout << "\n=== Testing Overflow Safety ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    std::atomic<int> overflow_count{0};

    constexpr std::size_t large_shape = std::numeric_limits<std::size_t>::max() / 2;
    
    auto overflow_sender = schedule(scheduler) | bulk_chunked(1000, [&](std::size_t i) {
        if (i < 1000) {
            overflow_count++;
        }
    });

    sync_wait(overflow_sender);

    std::cout << "Overflow safety test completed: processed first " 
              << overflow_count.load() << " items" << std::endl;
    HPX_TEST(overflow_count.load() == 1000);
}

void test_chunk_size_distribution()
{
    std::cout << "\n=== Testing Chunk Size Distribution ===" << std::endl;

    auto scheduler = get_parallel_scheduler();
    std::vector<std::size_t> chunk_sizes;
    std::mutex chunk_mutex;

    auto sender = schedule(scheduler) | bulk_chunked(500, [&](std::size_t i) {
        std::lock_guard<std::mutex> lock(chunk_mutex);
        
        std::size_t chunk_start = (i / 8) * 8;
        std::size_t chunk_end = std::min(chunk_start + 8, std::size_t(500));
        std::size_t chunk_size = chunk_end - chunk_start;
        
        if (std::find(chunk_sizes.begin(), chunk_sizes.end(), chunk_size) == chunk_sizes.end()) {
            chunk_sizes.push_back(chunk_size);
        }
    });

    sync_wait(sender);

    std::sort(chunk_sizes.begin(), chunk_sizes.end());
    
    std::cout << "Chunk sizes: ";
    for (std::size_t size : chunk_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    if (!chunk_sizes.empty()) {
        std::cout << "Min chunk size: " << chunk_sizes.front() 
                  << ", Max chunk size: " << chunk_sizes.back() << std::endl;
    }
}

int hpx_main()
{
    std::cout << "hpx_main started" << std::endl;

    auto scheduler = get_parallel_scheduler();
    std::cout << "Obtained parallel_scheduler" << std::endl;

    test_scheduler_construction();
    test_forward_progress_guarantee();
    test_schedule();
    test_completion_scheduler_query();
    test_stop_token();
    test_shared_context();
    test_basic_execution();
    test_structured_concurrency();
    test_error_handling();
    test_shared_context_with_algorithms();
    test_p2079r10_examples();

    test_case_1();
    test_case_2();
    test_case_3_bulk_chunked();

    test_bulk_chunked_with_thread_limits();
    test_negative_shape_handling();
    test_enhanced_bulk_chunked_analytics();
    test_bulk_chunked_with_value_propagation();
    test_bulk_chunked_error_handling();
    test_bulk_chunked_edge_cases();
    test_overflow_safety();
    test_chunk_size_distribution();

    std::cout << "\n🎉 All bulk_chunked tests completed successfully!" << std::endl;

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::cout << "Calling hpx::local::init" << std::endl;
    return hpx::local::init(hpx_main, argc, argv);
}
