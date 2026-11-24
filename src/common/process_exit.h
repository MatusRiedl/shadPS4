// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Common {

// Some DLLs crash in their process-detach code (NVIDIA's nvngx_dlssg.dll, loaded by Streamline for
// DLSS frame generation). Once one is in the process, it ends with TerminateProcess from the last
// exit handler, after the other handlers (emulator shutdown, log flush) ran, so DLL detach never
// runs. Windows only; elsewhere these only exit as usual.

/// First thing in main: its exit handlers then run last.
void InstallExitTerminator();

/// From now on the process ends with TerminateProcess. `before` (optional) runs first, once.
void SetTerminateOnExit(void (*before)() = nullptr);

/// std::quick_exit that keeps its exit code when the terminator ends the process.
[[noreturn]] void QuickExit(int code);

} // namespace Common
