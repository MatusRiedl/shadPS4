# NVIDIA Streamline headers

The public headers of NVIDIA Streamline SDK 2.14.16 (`include/`, MIT, see `LICENSE.txt`), for Overtime Native v3's
DLSS frame generation (`src/video_core/renderer_vulkan/host_passes/frame_gen.cpp`). Headers only: the Streamline and
NGX DLLs are not in this repository; players get them beside the exe.

Left out: `sl_nvperf.h`. One change: `sl_pcl.h` line 91, `using to_underlying = std::to_underlying;` (an alias of a
function template, which does not compile as C++23) is `using std::to_underlying;`.
