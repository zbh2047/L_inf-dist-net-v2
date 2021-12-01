#include "norm_dist.h"

const int BLOCK_SIZE = 16;
const int WARP_SIZE = 32;
const int MAX_BLOCK_CO_SIZE = 32;
const int MAX_BLOCK_B_SIZE = 16;

#define CONST_PTR const float* __restrict__
#define PTR float* __restrict__

#define EPS 1e-10f

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <bool has_hw, bool check_ci, bool ci_split, int GROUP_CI, int GROUP_CO, int GROUP_B> __global__
void inf_dist_forward_kernel(CONST_PTR input, CONST_PTR weight,
                             int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    int g = blockIdx.z % G, ci_split_id = blockIdx.z / G;
    int start_ci = ci_split_id * (CI_div_G / WARP_SIZE) / (gridDim.z / G) * WARP_SIZE;
    int end_ci = (ci_split_id + 1) * (CI_div_G / WARP_SIZE) / (gridDim.z / G) * WARP_SIZE;
    if (blockIdx.z / G == gridDim.z / G - 1) end_ci = CI_div_G;
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockI[GROUP_CI][GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[GROUP_CI][BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float max_output[GROUP_B][GROUP_CO];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++)
            max_output[i][j] = EPS;
    }

    for (int k = start_ci; k < end_ci; k += BLOCK_SIZE * GROUP_CI) {
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                if (b[i] < B) {
                    int channel = k + kk * BLOCK_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
                    int input_offset = ((b[i] * G + g) * CI_div_G + channel) * HW + hw[i];
                    if (check_ci) blockI[kk][i][threadIdx.y][threadIdx.x] = channel < end_ci ? input[input_offset] : 0;
                    else blockI[kk][i][threadIdx.y][threadIdx.x] = input[input_offset];
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CO; i++) {
                if (read_w_co + i * BLOCK_SIZE < CO_div_G) {
                    int channel = k + kk * BLOCK_SIZE + threadIdx.x;
                    int weight_offset = (g * CO_div_G + read_w_co + i * BLOCK_SIZE) * CI_div_G + channel;
                    if (check_ci) blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] =
                        channel < end_ci ? weight[weight_offset] : 0;
                    else blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = weight[weight_offset];
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int t = 0; t < BLOCK_SIZE; t++) {
                #pragma unroll
                for (int i = 0; i < GROUP_B; i++) {
                    #pragma unroll
                    for (int j = 0; j < GROUP_CO; j++) {
                        float x = has_hw ? blockI[kk][i][t][threadIdx.x] : blockI[kk][i][threadIdx.y][t];
                        float w = blockW[kk][t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                        max_output[i][j] = max(max_output[i][j], abs(x - w));
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        if (b[i] < B) {
            #pragma unroll
            for (int j = 0; j < GROUP_CO; j++) {
                int channel = write_co + j * BLOCK_SIZE;
                if (channel < CO_div_G) {
                    int offset = ((b[i] * G + g) * CO_div_G + channel) * HW + hw[i];
                    if (ci_split) atomicMax((int*)&output[offset], __float_as_int(max_output[i][j]));
                     // note that the result is always non-negative so such conversion is correct
                    else output[offset] = max_output[i][j];
                }
            }
        }
    }
}

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <bool has_hw, bool check_ci, int GROUP_CI, int GROUP_CO, int GROUP_B> __global__
void inf_dist_forward_kernel(CONST_PTR input, CONST_PTR weight,
                             int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output, int* __restrict__ pos) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    int g = blockIdx.z;
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockI[GROUP_CI][GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[GROUP_CI][BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float max_output[GROUP_B][GROUP_CO];
    int res_pos[GROUP_B][GROUP_CO];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            max_output[i][j] = EPS;
            res_pos[i][j] = 0;
        }
    }

    for (int k = 0; k < CI_div_G; k += BLOCK_SIZE * GROUP_CI) {
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                if (b[i] < B) {
                    int channel = k + kk * BLOCK_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
                    int input_offset = ((b[i] * G + g) * CI_div_G + channel) * HW + hw[i];
                    if (check_ci) blockI[kk][i][threadIdx.y][threadIdx.x] = channel < CI_div_G ? input[input_offset] : 0;
                    else blockI[kk][i][threadIdx.y][threadIdx.x] = input[input_offset];
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CO; i++) {
                if (read_w_co + i * BLOCK_SIZE < CO_div_G) {
                    int channel = k + kk * BLOCK_SIZE + threadIdx.x;
                    int weight_offset = (g * CO_div_G + read_w_co + i * BLOCK_SIZE) * CI_div_G + channel;
                    if (check_ci) blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] =
                        channel < CI_div_G ? weight[weight_offset] : 0;
                    else blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = weight[weight_offset];
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int t = 0; t < BLOCK_SIZE; t++) {
                #pragma unroll
                for (int i = 0; i < GROUP_B; i++) {
                    #pragma unroll
                    for (int j = 0; j < GROUP_CO; j++) {
                        float x = has_hw ? blockI[kk][i][t][threadIdx.x] : blockI[kk][i][threadIdx.y][t];
                        float w = blockW[kk][t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                        float tmp1 = x - w, tmp2 = abs(tmp1);
                        if (tmp2 > max_output[i][j]) {
                            max_output[i][j] = tmp2;
                            res_pos[i][j] = k + kk * BLOCK_SIZE + t + (tmp1 >= 0 ? 0 : 1 << 31);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        if (b[i] < B) {
            #pragma unroll
            for (int j = 0; j < GROUP_CO; j++) {
                int channel = write_co + j * BLOCK_SIZE;
                if (channel < CO_div_G) {
                    int offset = ((b[i] * G + g) * CO_div_G + channel) * HW + hw[i];
                    output[offset] = max_output[i][j];
                    pos[offset] = res_pos[i][j];
                }
            }
        }
    }
}

template <bool has_hw, int BLOCK_CO_SIZE, int BLOCK_B_SIZE>
__global__ void inf_dist_backward_input_kernel(CONST_PTR grad_output, const int* __restrict__ pos,
                                                int B, int CO_div_G, int CI_div_G, int HW, int G, PTR grad_input) {
    #pragma unroll
    for (int j = 0; j < BLOCK_B_SIZE; j += BLOCK_SIZE){
        int b_hw = blockIdx.x * BLOCK_B_SIZE + j + (has_hw ? threadIdx.x : threadIdx.y);
        int b = has_hw ? b_hw / HW : b_hw;
        int hw = has_hw ? b_hw % HW : 0;
        int co = blockIdx.y * BLOCK_CO_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
        int offset = ((b * G + blockIdx.z) * CO_div_G + co) * HW + hw;
        #pragma unroll
        for (int i = 0; i < BLOCK_CO_SIZE; i += BLOCK_SIZE){
            if (b < B && co + i < CO_div_G) {
                int pos_reg = pos[offset + i * HW];
                float grad = grad_output[offset + i * HW];
                int index = pos_reg & (~(1 << 31));
                float value = pos_reg >= 0 ? grad : -grad;
                atomicAdd(&grad_input[((b * G + blockIdx.z) * CI_div_G + index) * HW + hw], value);
            }
        }
    }
}

template <bool has_hw, int BLOCK_CO_SIZE, int BLOCK_B_SIZE>
__global__ void inf_dist_backward_input_weight_kernel(CONST_PTR grad_output, const int* __restrict__ pos,
                                                      int B, int CO_div_G, int CI_div_G, int HW, int G,
                                                      PTR grad_input, PTR grad_weight) {
    #pragma unroll
    for (int j = 0; j < BLOCK_B_SIZE; j += BLOCK_SIZE){
        int b_hw = blockIdx.x * BLOCK_B_SIZE + j + (has_hw ? threadIdx.x : threadIdx.y);
        int b = has_hw ? b_hw / HW : b_hw;
        int hw = has_hw ? b_hw % HW : 0;
        int co = blockIdx.y * BLOCK_CO_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
        int offset = ((b * G + blockIdx.z) * CO_div_G + co) * HW + hw;
        #pragma unroll
        for (int i = 0; i < BLOCK_CO_SIZE; i += BLOCK_SIZE){
            if (b < B && co + i < CO_div_G) {
                int pos_reg = pos[offset + i * HW];
                float grad = grad_output[offset + i * HW];
                int index = pos_reg & (~(1 << 31));
                float value = pos_reg >= 0 ? grad : -grad;
                atomicAdd(&grad_input[((b * G + blockIdx.z) * CI_div_G + index) * HW + hw], value);
                atomicAdd(&grad_weight[(blockIdx.z * CO_div_G + co + i) * CI_div_G + index], -value);
            }
        }
    }
}

#define inf_dist_forward_nograd_helper_func(func, GROUP_CO, GROUP_B, split_ci, has_hw, paras...) \
    int num_block_co = (CO / G - 1) / (BLOCK_SIZE * GROUP_CO) + 1; \
    int num_block_b = (B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1; \
    int CI_div_G = CI / G; \
    int ci_split = 1; \
    if (split_ci) { \
        cudaDeviceProp deviceProp; \
        cudaGetDeviceProperties(&deviceProp, 0); \
        int ci_split1 = max(deviceProp.multiProcessorCount * 36 / (num_block_co * num_block_b * G), 1); \
        ci_split1 = ci_split1 <= 2 ? ci_split1 : ci_split1 <= 6 ? 4 : 8; \
        ci_split = min(ci_split1, (CI_div_G - 1) / BLOCK_SIZE + 1); \
    } \
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); \
    dim3 dimGrid(num_block_b, num_block_co, G * ci_split); \
    if (CI_div_G % (1 * BLOCK_SIZE * ci_split) == 0) { \
        if (ci_split == 1) func<has_hw, false, false, 1, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        else { \
            cudaMemset(output, 0, B * CO * HW * sizeof(float)); \
            func<has_hw, false, true, 1, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        } \
    } \
    else { \
        if (ci_split == 1) func<has_hw, true, false, 1, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        else { \
            cudaMemset(output, 0, B * CO * HW * sizeof(float)); \
            func<has_hw, true, true, 1, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        } \
    }

void inf_dist::forward_nograd(const float* input, const float* weight,
                              int B, int CO, int CI, int G, int HW, float* output) {
    if (HW == 1) {
        inf_dist_forward_nograd_helper_func(inf_dist_forward_kernel, 4, 4, true, false,
                                            input, weight, B, CO / G, CI / G, HW, G, output);
    }
    else if (CO / G <= 2 * BLOCK_SIZE) {
        inf_dist_forward_nograd_helper_func(inf_dist_forward_kernel, 2, 4, false, true,
                                            input, weight, B, CO / G, CI / G, HW, G, output);
    }
    else {
        inf_dist_forward_nograd_helper_func(inf_dist_forward_kernel, 4, 4, false, true,
                                            input, weight, B, CO / G, CI / G, HW, G, output);
    }
}

#define inf_dist_forward_helper_func(func, GROUP_CO, GROUP_B, has_hw, paras...) \
    int num_block_co = (CO / G - 1) / (BLOCK_SIZE * GROUP_CO) + 1; \
    int num_block_b = (B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1; \
    int CI_div_G = CI / G; \
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); \
    dim3 dimGrid(num_block_b, num_block_co, G); \
    if (CI_div_G % (1 * BLOCK_SIZE) == 0) \
        func<has_hw, false, 1, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
    else func<has_hw, true, 1, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras);

void inf_dist::forward(const float* input, const float* weight,
                       int B, int CO, int CI, int G, int HW, float* output, int* pos) {
    if (HW == 1) {
        inf_dist_forward_helper_func(inf_dist_forward_kernel, 4, 4, false,
                                     input, weight, B, CO / G, CI / G, HW, G, output, pos);
    }
    else if (CO / G <= 2 * BLOCK_SIZE) {
        inf_dist_forward_helper_func(inf_dist_forward_kernel, 2, 4, true,
                                     input, weight, B, CO / G, CI / G, HW, G, output, pos);
    }
    else {
        inf_dist_forward_helper_func(inf_dist_forward_kernel, 4, 4, true,
                                     input, weight, B, CO / G, CI / G, HW, G, output, pos);
    }
}

#define inf_dist_backward_helper_func(func, HW, CO_div_G, paras...) \
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); \
    int num_block_b = (B * HW - 1) / MAX_BLOCK_B_SIZE + 1; \
    if (HW == 1) { \
        dim3 dimGrid2(num_block_b, (CO_div_G - 1) / BLOCK_SIZE + 1, G); \
        func<false, MAX_BLOCK_CO_SIZE, MAX_BLOCK_B_SIZE><<<dimGrid2, dimBlock>>>(paras); \
    } \
    else { \
        dim3 dimGrid2(num_block_b, (CO_div_G - 1) / (2 * BLOCK_SIZE) + 1, G); \
        func<true, MAX_BLOCK_CO_SIZE, MAX_BLOCK_B_SIZE><<<dimGrid2, dimBlock>>>(paras); \
    }

void inf_dist::backward_input(const float* grad_output, const int* pos,
                              int B, int CO, int CI, int G, int HW, float* grad_input) {
    int CO_div_G = CO / G;
    cudaMemset(grad_input, 0, B * CI * HW * sizeof(float));
    inf_dist_backward_helper_func(inf_dist_backward_input_kernel, HW, CO_div_G,
                                  grad_output, pos, B, CO_div_G, CI / G, HW, G, grad_input);
}

void inf_dist::backward_input_weight(const float* grad_output, const int* pos,
                                     int B, int CO, int CI, int G, int HW, float* grad_input, float* grad_weight) {
    int CO_div_G = CO / G;
    cudaMemset(grad_input, 0, B * CI * HW * sizeof(float));
    cudaMemset(grad_weight, 0, CO * (CI / G) * sizeof(float));
    inf_dist_backward_helper_func(inf_dist_backward_input_weight_kernel, HW, CO_div_G,
                                  grad_output, pos, B, CO_div_G, CI / G, HW, G, grad_input, grad_weight);
}
