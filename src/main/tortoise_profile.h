#ifndef _TORTOISE_PROFILE_H
#define _TORTOISE_PROFILE_H

// Coarse wall-clock stage timers. Each scope prints one line to stderr on exit:
//     [PROFILE] <name> <seconds>
// Summing them per run attributes wall time to named stages (plan_optimize.md M0).
// Measurement only: no arithmetic, no allocation, no synchronisation.

#include <chrono>
#include <cstdio>
#include <string>

struct ProfileScope
{
    std::string name;
    std::chrono::steady_clock::time_point t0;

    explicit ProfileScope(const std::string &n)
        : name(n), t0(std::chrono::steady_clock::now()) {}

    ~ProfileScope()
    {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[PROFILE] %s %.3f\n", name.c_str(), s);
        fflush(stderr);
    }
};

#define TORTOISE_PROFILE(nm) ProfileScope profile_scope_(nm)

#endif
