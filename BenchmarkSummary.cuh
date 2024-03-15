#ifndef BENCHMARK_H
#define BENCHMARK_H

void gpuAssert(cudaError_t code, const char *file, int line, bool abort);
#define gpuErrChk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__, true); \
	}

#include <stdio.h>
#include <string>
#include <unordered_map>
#include <cassert>
#include <cuda_runtime.h>

class KernelInfo
{
public:
    std::string name; // name of the kernel
    float time; // total time spent in the kernel
    int count; // number of times the kernel was called
    KernelInfo(std::string name) : name(name), time(0.0f), count(0) {}
    KernelInfo() : name(""), time(0.0f), count(0) {}
};

#define MAX_CUDA_DEVICES 8

// Singleton class for summarizing the results of the benchmark
class BenchmarkSummary_cpp
{
private:

    // Making the constructor private for singleton
    BenchmarkSummary_cpp() {}
    
    static std::unordered_map<std::string, KernelInfo> kernelInfos; // hashmap of kernel info

    // hashmap of cuda timers, two (start, end) for each kernel
    // static std::unordered_map<std::string, std::pair<cudaEvent_t, cudaEvent_t>> kernelTimers;
    static std::array<std::unordered_map<std::string, std::pair<cudaEvent_t, cudaEvent_t>>, MAX_CUDA_DEVICES> kernelTimers;

public:

    // Deleted copy constructor and assignment operator for singleton
    BenchmarkSummary_cpp(const BenchmarkSummary_cpp&) = delete;
    BenchmarkSummary_cpp& operator=(const BenchmarkSummary_cpp&) = delete;

    // Static method for accessing class instance
    static BenchmarkSummary_cpp& getInstance() {
        static BenchmarkSummary_cpp instance;
        return instance;
    }

    static void addKernelType(std::string name, int device)
    {
        assert(kernelTimers[device].find(name) == kernelTimers[device].end());
        if(kernelInfos.find(name) == kernelInfos.end())
        {
            kernelInfos[name] = KernelInfo(name);
        }
        gpuErrChk(cudaEventCreate(&kernelTimers[device][name].first));
        gpuErrChk(cudaEventCreate(&kernelTimers[device][name].second));
    }

    static void recordStart(std::string name)
    {
        int device;
        gpuErrChk(cudaGetDevice(&device));

        if(kernelTimers[device].find(name) == kernelTimers[device].end())
        {
            addKernelType(name, device);
        }
        gpuErrChk(cudaEventRecord(kernelTimers[device][name].first));
    }

    static void recordEnd(std::string name)
    {
        int device;
        gpuErrChk(cudaGetDevice(&device));

        gpuErrChk(cudaEventRecord(kernelTimers[device][name].second));
        gpuErrChk(cudaEventSynchronize(kernelTimers[device][name].second));
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, kernelTimers[device][name].first, kernelTimers[device][name].second);
        kernelInfos[name].time += milliseconds;
        kernelInfos[name].count++;
    }

    static void printSummary()
    {
        printf("#############################################\n");
        printf("Benchmark summary:\n");
        printf("#############################################\n");

        float sum_all_times = 0;
        for (auto &it : kernelInfos)
        {
            sum_all_times += it.second.time;
        }
        for (auto &it : kernelInfos)
        {
            printf("%s: %f s (%d calls), avg: %f ms, proportion: %.2f\%\n",
                it.second.name.c_str(), it.second.time / 1000.0f, it.second.count, it.second.time / it.second.count, it.second.time / sum_all_times * 100);
        }
        printf("\n");
    }

    static float getAverageTime(std::string name)
    {
        return kernelInfos[name].time / kernelInfos[name].count;
    }

    static float getTotalTime(std::string name)
    {
        return kernelInfos[name].time;
    }

    static void destroy()
    {
        for (auto &it : kernelTimers)
        {
            for (auto &it2 : it)
            {
                gpuErrChk(cudaEventDestroy(it2.second.first));
                gpuErrChk(cudaEventDestroy(it2.second.second));
            }
        }
    }

    ~BenchmarkSummary_cpp()
    {
        destroy();
    }
};

std::unordered_map<std::string, KernelInfo> BenchmarkSummary_cpp::kernelInfos;
std::array<std::unordered_map<std::string, std::pair<cudaEvent_t, cudaEvent_t>>, MAX_CUDA_DEVICES> BenchmarkSummary_cpp::kernelTimers;

// C interface for the singleton class
extern "C"
{
    void addKernelType(char name[])
    {
        int device;
        gpuErrChk(cudaGetDevice(&device));

        std::string name_str(name);

        BenchmarkSummary_cpp::getInstance().addKernelType( name_str, device);
    }

    void recordStart(char name[])
    {
        std::string name_str(name);
        BenchmarkSummary_cpp::getInstance().recordStart(name_str);
    }

    void recordEnd(char name[])
    {
        std::string name_str(name);
        BenchmarkSummary_cpp::getInstance().recordEnd(name_str);
    }

    void printSummary()
    {
        BenchmarkSummary_cpp::getInstance().printSummary();
    }

    float getAverageTime(char name[])
    {
        std::string name_str(name);
        return BenchmarkSummary_cpp::getInstance().getAverageTime(name_str);
    }

    float getTotalTime(char name[])
    {
        std::string name_str(name);
        return BenchmarkSummary_cpp::getInstance().getTotalTime(name_str);
    }
}

#endif
