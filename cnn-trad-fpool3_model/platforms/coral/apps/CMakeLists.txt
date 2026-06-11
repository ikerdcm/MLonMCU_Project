# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_subdirectory(kws_coral)
add_subdirectory(kws_live)
add_subdirectory(kws_bench)
add_subdirectory(single_inference)
add_subdirectory(kws_bench_cpu)   # fp32-cpu offline (no Edge TPU)
add_subdirectory(kws_idle_cpu)    # fp32-cpu offline duty-cycle (1 inf / 5 s, WFI sleep) — power
add_subdirectory(kws_live_cpu)    # fp32-cpu online  (no Edge TPU)
add_subdirectory(kws_eval)        # on-device accuracy (device-in-the-loop, v2 int8)
add_subdirectory(kws_eval_cpu)    # on-device accuracy, fp32-cpu (M7, no Edge TPU)
add_subdirectory(kws_eval_stream)     # streaming eval: raw audio -> frontend -> Edge TPU (testbench.py)
add_subdirectory(kws_eval_stream_cpu) # streaming eval, fp32 CPU (M7, no Edge TPU)
