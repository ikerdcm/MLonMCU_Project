/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>

#include "CycleCounter.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"

// RW: Remove MAINSTACKSIZE because gap9-sdk does not use it
#define SLAVESTACKSIZE    3800
#define N_POWER_RUNS      10
// Sleep between inference peaks so each lands in its own 5-second PPK2 window.
// Inference ~117ms + POWER_SLEEP_MS ≈ 5000ms per window.
#define POWER_SLEEP_MS    4880

struct pi_device cluster_dev;
uint32_t total_cycles = 0;

typedef struct {
  void *expected;
  void *actual;
  int num_elements;
  int output_buf_index;
  int *err_count;
} FloatCompareArgs;

void CompareFloatOnCluster(void *args) {

  if (pi_core_id() == 0) {
    FloatCompareArgs *compare_args = (FloatCompareArgs *)args;
    float *expected = (float *)compare_args->expected;
    float *actual = (float *)compare_args->actual;
    int num_elements = compare_args->num_elements;
    int output_buf_index = compare_args->output_buf_index;
    int *err_count = compare_args->err_count;

    int local_err_count = 0;

    for (int i = 0; i < num_elements; i++) {
      float expected_val = expected[i];
      float actual_val = actual[i];
      float diff = expected_val - actual_val;

      if ((diff < -1e-4) || (diff > 1e-4) || isnan(diff)) {
        local_err_count += 1;

        printf("Expected: %10.6f  ", expected_val);
        printf("Actual: %10.6f  ", actual_val);
        printf("Diff: %10.6f at Index %12u in Output %u\r\n", diff, i,
               output_buf_index);
      }
    }

    *err_count = local_err_count;
  }
}

void CL_CompareFloat(void *arg) {
  pi_cl_team_fork(NUM_CORES, CompareFloatOnCluster, arg);
}

void InitNetworkWrapper(void *args) {
  (void)args;
  InitNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

void RunNetworkWrapper(void *args) {
  (void)args;
  ResetTimer();
  StartTimer();
  RunNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
  total_cycles = getCycles();
  StopTimer();
}

int main(void) {

  struct pi_cluster_conf conf;
  pi_cluster_conf_init(&conf);
  conf.id = 0;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;

  pi_freq_set(PI_FREQ_DOMAIN_FC, 240000000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 240000000);

#ifndef NOFLASH
  mem_init();
  open_fs();
#endif

  printf("Initializing\n");

  struct pi_cluster_task cluster_task;

  pi_cluster_task(&cluster_task, InitNetworkWrapper, NULL);
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

#ifndef CI
  printf("Initialized\r\n");
#endif
  for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    if ((uint32_t)DeeployNetwork_inputs[buf] >= 0x10000000) {
      memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
             DeeployNetwork_inputs_bytes[buf]);
    }
  }

  // N_POWER_RUNS consecutive inferences — cluster stays open, no valley between peaks.
  // RunNetwork frees the input buffer on each call; re-alloc + re-copy before each repeat.
  for (int run = 0; run < N_POWER_RUNS; run++) {
    if (run > 0) {
      for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
        DeeployNetwork_inputs[buf] = pi_l2_malloc(DeeployNetwork_inputs_bytes[buf]);
        memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
               DeeployNetwork_inputs_bytes[buf]);
      }
    }
    pi_cluster_task(&cluster_task, RunNetworkWrapper, NULL);
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    printf("Runtime: %u cycles\r\n", total_cycles);
    vTaskDelay(pdMS_TO_TICKS(POWER_SLEEP_MS));
  }

  uint32_t tot_err, tot_tested;
  tot_err = 0;
  tot_tested = 0;
  void *compbuf;
  FloatCompareArgs float_compare_args;
  uint32_t float_error_count = 0;

  for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
    tot_tested += DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE);

    if ((uint32_t)DeeployNetwork_outputs[buf] < 0x10000000) {
      compbuf = pi_l2_malloc(DeeployNetwork_outputs_bytes[buf]);
      ram_read(compbuf, DeeployNetwork_outputs[buf],
               DeeployNetwork_outputs_bytes[buf]);
    } else {
      compbuf = DeeployNetwork_outputs[buf];
    }

    if (ISOUTPUTFLOAT) {
      float_error_count = 0;
      float_compare_args.expected = testOutputVector[buf];
      float_compare_args.actual = compbuf;
      float_compare_args.num_elements =
          DeeployNetwork_outputs_bytes[buf] / sizeof(float);
      float_compare_args.output_buf_index = buf;
      float_compare_args.err_count = (int *)&float_error_count;

      pi_cluster_task(&cluster_task, CL_CompareFloat, &float_compare_args);
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      tot_err += float_error_count;
    } else {

      for (uint32_t i = 0;
           i < DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE); i++) {
        OUTPUTTYPE expected = ((OUTPUTTYPE *)testOutputVector[buf])[i];
        OUTPUTTYPE actual = ((OUTPUTTYPE *)compbuf)[i];
        OUTPUTTYPE diff = expected - actual;

        if (diff) {
          tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t)DeeployNetwork_outputs[buf] < 0x10000000) {
      pi_l2_free(compbuf, DeeployNetwork_outputs_bytes[buf]);
    }
  }

  printf("Errors: %u out of %u \r\n", tot_err, tot_tested);

  // Print predicted class (argmax over float output buffer 0)
  if (ISOUTPUTFLOAT && DeeployNetwork_num_outputs > 0) {
    float *out = (float *)DeeployNetwork_outputs[0];
    int n = DeeployNetwork_outputs_bytes[0] / sizeof(float);
    int argmax = 0;
    for (int i = 1; i < n; i++)
      if (out[i] > out[argmax]) argmax = i;
    printf("Predicted class: %d (confidence %.4f)\r\n", argmax, out[argmax]);
  }

  // Print predicted class (argmax over integer/quantized output buffer 0)
  if (!ISOUTPUTFLOAT && DeeployNetwork_num_outputs > 0) {
    OUTPUTTYPE *out = (OUTPUTTYPE *)DeeployNetwork_outputs[0];
    int n = DeeployNetwork_outputs_bytes[0] / sizeof(OUTPUTTYPE);
    int argmax = 0;
    for (int i = 1; i < n; i++)
      if (out[i] > out[argmax]) argmax = i;
    printf("Predicted class: %d (logit %d)\r\n", argmax, (int)out[argmax]);
  }

  return 0;
}
