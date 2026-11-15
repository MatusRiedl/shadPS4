// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Overtime switches. Each is on unless its environment variable is "0":
//   OVERTIME_LAZY_COMMIT  (Windows) guest memory backing reserved, committed range by range when
//                         first mapped, instead of the whole backing committed at start
//   OVERTIME_FLIP_TOLERANT  a flip IRQ whose buffer label was already reset logs a warning and
//                         flips, instead of the "Out of order flip IRQ" assert (crash)
//   OVERTIME_POLL_THREAD  controllers (4 ms) and mouse (32 ms) polled from a sleeping thread
//                         instead of SDL timers (SDL's timer thread spins before each tick)
//   OVERTIME_GC_SECONDS   the texture collector ages images in seconds (20 / 5 / 1) instead of
//                         guest submissions, and cards under 8 GB get its marks at 70 / 85 / 95 %
//                         of the driver's current budget (the rule of thegh0sts' native port)
// Overtime Native v2 (the startup line "Overtime Native v2 switches" logs each state):
//   OVERTIME_RB_GUEST     a readback whose GPU writes were all submitted copies on a second queue,
//                         waiting only for the last write (on the faulting guest thread when it
//                         can), instead of the GPU thread draining the whole GPU (Kravickas)
//   OVERTIME_LOOKUP_MEMO  draws reuse the last render target, depth and texture lookup when the
//                         registers / T# and the texture cache registry are unchanged (Kravickas)
//   OVERTIME_STREAM_REUSE a read-only binding equal to an earlier copy of the same range in the
//                         same command buffer reuses that copy (Kravickas)
//   OVERTIME_THREAD_PRIORITY  game Main/Sim/Render/Audio threads above normal, GPU command
//                         processor highest, present thread above normal (Windows)
//   OVERTIME_ASYNC_PIPELINES  a new graphics pipeline of a busy frame is built on worker threads;
//                         its draws are skipped until it is ready
//   OVERTIME_VK_RECORDER  draw commands are recorded into the Vulkan command buffer by a second
//                         thread while the GPU thread processes the next draws

#pragma once

#include <cstdlib>

namespace Overtime {

inline bool SwitchOn(const char* name) {
    const char* v = std::getenv(name);
    return !(v && v[0] == '0');
}

} // namespace Overtime
