// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/process_exit.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "common/logging/log.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Common {

namespace {

std::atomic<bool> g_terminate{false};
std::atomic<void (*)()> g_before{nullptr};
std::atomic<int> g_code{0};

void Terminator() {
#ifdef _WIN32
    if (!g_terminate.load()) {
        return;
    }
    if (auto* before = g_before.exchange(nullptr)) {
        before();
    }
    Log::Flush();
    std::fflush(stdout);
    std::fflush(stderr);
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(g_code.load()));
#endif
}

} // namespace

void InstallExitTerminator() {
    static std::atomic<bool> installed{false};
    if (installed.exchange(true)) {
        return;
    }
    std::atexit(Terminator);
    std::at_quick_exit(Terminator);
}

void SetTerminateOnExit(void (*before)()) {
    if (before != nullptr) {
        g_before.store(before);
    }
    g_terminate.store(true);
}

void QuickExit(int code) {
    g_code.store(code);
    std::quick_exit(code);
}

} // namespace Common
