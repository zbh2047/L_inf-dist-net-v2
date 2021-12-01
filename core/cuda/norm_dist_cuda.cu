#include "fast_pow.cuh"
#include "norm_dist.h"

const int BLOCK_SIZE = 16;
const int BLOCK_CI_SIZE = 8;
const int WARP_SIZE = 32;
const int MAX_BLOCK_CO_SIZE = 64;
const int BATCH_SPLIT = 4;

#define CONST_PTR const float* __restrict__
#define PTR float* __restrict__

#define EPS 1e-10f
#define EPS2 1e-6f
#define UNDER_FLOW_EPS 1.175494351e-38f

template<int ip> __device__ __forceinline__
float update_forward(float x, float w, float p) {
    float t = x - w;
    return pow_fun<ip, false>(t, p);
}

template<int ip> __device__ __forceinline__
float update_forward(float x, float w, float p, float r_max_x_sub_w) {
    float t = x - w;
    return pow_fun<ip, false>(t * r_max_x_sub_w, p);
}

__device__ __forceinline__
void normalize(float& output_reg, float ratio, float p) {
    output_reg = output_reg * pow_fun(ratio, -p);
}

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <int ip, bool has_hw, bool check_ci, int GROUP_CI, int GROUP_CO, int GROUP_B> __global__
void norm_dist_forward_kernel(CONST_PTR input, CONST_PTR weight,
                              int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output, float p) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
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

    float res[GROUP_B][GROUP_CO];
    float max_output[GROUP_B][GROUP_CO], r_max_output[GROUP_B][GROUP_CO];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            res[i][j] = 1.0f;
            max_output[i][j] = EPS;
            r_max_output[i][j] = 1.0f / EPS;
        }
    }

    for (int k = 0; k < CI_div_G; k += BLOCK_SIZE * GROUP_CI) {
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                if (b[i] < B) {
                    int channel = k + kk * BLOCK_SIZE + (has_hw ? threadIdx.y : threadIdx.x);
                    int input_offset = ((b[i] * G + blockIdx.z) * CI_div_G + channel) * HW + hw[i];
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
                    int weight_offset = (blockIdx.z * CO_div_G + read_w_co + i * BLOCK_SIZE) * CI_div_G + channel;
                    if (check_ci) blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] =
                        channel < CI_div_G ? weight[weight_offset] : 0;
                    else blockW[kk][threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = weight[weight_offset];
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < GROUP_CI; kk++) {
            #pragma unroll(4)
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
        #pragma unroll
        for (int i = 0; i < GROUP_B; i++) {
            #pragma unroll
            for (int j = 0; j < GROUP_CO; j++) {
                float ratio = max_output[i][j] * r_max_output[i][j];
                if (ratio > 1.0f + EPS2) {
                    normalize(res[i][j], ratio, p);
                    r_max_output[i][j] = __frcp_rn(max_output[i][j]);
                }
            }
        }
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
                        res[i][j] += update_forward<ip>(x, w, p, r_max_output[i][j]);
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
                    float ans = pow_fun(res[i][j], __frcp_rn(p)) * max_output[i][j];
                    output[((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i]] = ans;
                }
            }
        }
    }
}

// To maximize performance, adjust GROUP_CO, GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <int ip, bool has_hw, bool has_max, bool check_ci, int GROUP_CO, int GROUP_B> __global__
void norm_dist_forward_kernel(CONST_PTR input, CONST_PTR weight, CONST_PTR max_output,
                              int B, int CO_div_G, int CI_div_G, int HW, int G, PTR output, float p) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }

    int write_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_co = blockIdx.y * (BLOCK_SIZE * GROUP_CO) + threadIdx.y;

    __shared__ float blockI[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CI * B if has_hw else B * CI
    __shared__ float blockW[BLOCK_SIZE][BLOCK_SIZE * GROUP_CO + WARP_SIZE / BLOCK_SIZE]; // CI * CO

    float res[GROUP_B][GROUP_CO];
    float r_max_output[GROUP_B][GROUP_CO];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CO; j++) {
            if (has_max) {
                int channel = write_co + j * BLOCK_SIZE;
                if (b[i] < B && channel < CO_div_G)
                    r_max_output[i][j] = __frcp_rn(max_output[((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i]]);
                res[i][j] = pow_fun<ip, true>(EPS * r_max_output[i][j], p);
            }
            else res[i][j] = 0;
        }
    }

    for (int k = 0; k < CI_div_G; k += BLOCK_SIZE) {
        #pragma unroll
        for (int i = 0; i < GROUP_B; i++) {
            if (b[i] < B) {
                int channel = k + (has_hw ? threadIdx.y : threadIdx.x);
                int input_offset = ((b[i] * G + blockIdx.z) * CI_div_G + channel) * HW + hw[i];
                if (check_ci) blockI[i][threadIdx.y][threadIdx.x] = channel < CI_div_G ? input[input_offset] : 0;
                else blockI[i][threadIdx.y][threadIdx.x] = input[input_offset];
            }
        }
        #pragma unroll
        for (int i = 0; i < GROUP_CO; i++) {
            if (read_w_co + i * BLOCK_SIZE < CO_div_G) {
                int channel = k + threadIdx.x;
                int weight_offset = (blockIdx.z * CO_div_G + read_w_co + i * BLOCK_SIZE) * CI_div_G + channel;
                if (check_ci) blockW[threadIdx.x][threadIdx.y + i * BLOCK_SIZE] =
                    channel < CI_div_G ? weight[weight_offset] : 0;
                else blockW[threadIdx.x][threadIdx.y + i * BLOCK_SIZE] = weight[weight_offset];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE; t++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                #pragma unroll
                for (int j = 0; j < GROUP_CO; j++) {
                    float x = has_hw ? blockI[i][t][threadIdx.x] : blockI[i][threadIdx.y][t];
                    float w = blockW[t][(has_hw ? threadIdx.y : threadIdx.x) + j * BLOCK_SIZE];
                    if (has_max) res[i][j] += update_forward<ip>(x, w, p, r_max_output[i][j]);
                    else res[i][j] += update_forward<ip>(x, w, p);
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
                    float ans;
                    if (has_max) ans = __fdiv_rn(pow_fun(res[i][j], __frcp_rn(p)), r_max_output[i][j]);
                    else ans = pow_fun(max(res[i][j], UNDER_FLOW_EPS), __frcp_rn(p)); // for underflow case
                    output[((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i]] = ans;
                }
            }
        }
    }
}

template <int ip> __device__ __forceinline__
float update_backward_input(float x, float w, float r_o, float g, float p) {
    float t = x - w;
    if ((ip & 1) && ip >= 0) return g * pow_fun<ip, true>(t * r_o, p);
    return g * pow_fun<ip, false>(t * r_o, p) * (t >= 0 ? 1 : -1);
}

// To maximize performance, adjust GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <int ip, bool has_hw, bool check_co, int GROUP_CI, int GROUP_B> __global__
void norm_dist_backward_input_kernel(CONST_PTR grad_output, CONST_PTR input, CONST_PTR weight, CONST_PTR output,
                                     int B, int CO_div_G, int CI_div_G, int HW, int G, PTR grad_input, float p) {
    int b_hw = blockIdx.x * (BLOCK_SIZE * GROUP_B) + (has_hw ? threadIdx.x : threadIdx.y);
    int b[GROUP_B], hw[GROUP_B];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        int b_hw_i = b_hw + i * BLOCK_SIZE;
        if (has_hw) { b[i] = b_hw_i / HW; hw[i] = b_hw_i % HW; }
        else { b[i] = b_hw_i; hw[i] = 0; }
    }

    int write_ci = blockIdx.y * (BLOCK_SIZE * GROUP_CI) + (has_hw ? threadIdx.y : threadIdx.x);
    int read_w_ci = blockIdx.y * (BLOCK_SIZE * GROUP_CI) + threadIdx.x;

    __shared__ float blockO[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CO * B if has_hw else B * CO
    __shared__ float blockG[GROUP_B][BLOCK_SIZE][BLOCK_SIZE]; // CO * B if has_hw else B * CO
    __shared__ float blockW[GROUP_CI][BLOCK_SIZE][BLOCK_SIZE]; // CO * CI

    p -= 1;

    float res[GROUP_B][GROUP_CI], x[GROUP_B][GROUP_CI];
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        #pragma unroll
        for (int j = 0; j < GROUP_CI; j++) {
            res[i][j] = 0;
            if (b[i] < B && write_ci + j * BLOCK_SIZE < CI_div_G)
                x[i][j] = input[((b[i] * G + blockIdx.z) * CI_div_G + write_ci + j * BLOCK_SIZE) * HW + hw[i]];
            else x[i][j] = 0;
        }
    }

    for (int k = 0; k < CO_div_G; k += BLOCK_SIZE) {
        #pragma unroll
        for (int i = 0; i < GROUP_B; i++) {
            if (b[i] < B) {
                int channel = k + (has_hw ? threadIdx.y : threadIdx.x);
                int output_offset = ((b[i] * G + blockIdx.z) * CO_div_G + channel) * HW + hw[i];
                if (check_co) {
                    blockO[i][threadIdx.y][threadIdx.x] = channel < CO_div_G ? __frcp_rn(output[output_offset]) : 0;
                    blockG[i][threadIdx.y][threadIdx.x] = channel < CO_div_G ? grad_output[output_offset] : 0;
                }
                else {
                    blockO[i][threadIdx.y][threadIdx.x] = __frcp_rn(output[output_offset]);
                    blockG[i][threadIdx.y][threadIdx.x] = grad_output[output_offset];
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++) {
            if (read_w_ci + i * BLOCK_SIZE < CI_div_G) {
                int channel = k + threadIdx.y;
                int weight_offset = (blockIdx.z * CO_div_G + channel) * CI_div_G + read_w_ci + i * BLOCK_SIZE;
                if (check_co) blockW[i][threadIdx.y][threadIdx.x] =
                    channel < CO_div_G ? weight[weight_offset] : 0;
                else blockW[i][threadIdx.y][threadIdx.x] = weight[weight_offset];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE; t++) {
            #pragma unroll
            for (int i = 0; i < GROUP_B; i++) {
                #pragma unroll
                for (int j = 0; j < GROUP_CI; j++) {
                    float r_o = has_hw ? blockO[i][t][threadIdx.x] : blockO[i][threadIdx.y][t];
                    float g = has_hw ? blockG[i][t][threadIdx.x] : blockG[i][threadIdx.y][t];
                    float w = blockW[j][t][has_hw ? threadIdx.y : threadIdx.x];
                    res[i][j] += update_backward_input<ip>(x[i][j], w, r_o, g, p);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < GROUP_B; i++) {
        if (b[i] < B) {
            #pragma unroll
            for (int j = 0; j < GROUP_CI; j++) {
                int channel = write_ci + j * BLOCK_SIZE;
                if (channel < CI_div_G) {
                    grad_input[((b[i] * G + blockIdx.z) * CI_div_G + channel) * HW + hw[i]] = res[i][j];
                }
            }
        }
    }
}

// To maximize performance, adjust GROUP_B, GROUP_CI and #pragma unroll count for variable t
template <int ip, bool has_hw, bool multiple_co_block, int GROUP_CI, int GROUP_B, int BLOCK_CO_SIZE> __global__
void norm_dist_backward_input_weight_kernel(CONST_PTR grad_output, CONST_PTR input, CONST_PTR weight, CONST_PTR output,
                                            int B, int CO_div_G, int CI_div_G, int HW, int G,
                                            PTR grad_input, PTR grad_weight, float p, int num_block_ci) {
    static_assert(GROUP_B == 1 || GROUP_B == 2);
    __shared__ float blockO[WARP_SIZE][WARP_SIZE]; // has_hw ? CO * B : B * CO
    __shared__ float blockGO[WARP_SIZE][WARP_SIZE]; // has_hw ? CO * B : B * CO
    __shared__ float blockI[BLOCK_CI_SIZE * GROUP_CI][WARP_SIZE + WARP_SIZE / BLOCK_CI_SIZE]; // CI * B
    __shared__ float blockW[BLOCK_CI_SIZE * GROUP_CI][BLOCK_CO_SIZE + WARP_SIZE / BLOCK_CI_SIZE]; // CI * CO

    int start_block_CI = blockIdx.x % num_block_ci * BLOCK_CI_SIZE * GROUP_CI;
    int start_block_CO = blockIdx.x / num_block_ci * BLOCK_CO_SIZE;
    int batch_split = min(BATCH_SPLIT, HW), tot_block_BHW = B * batch_split;
    int start_block_BHW = blockIdx.y * tot_block_BHW / gridDim.y * HW / batch_split;
    int end_block_BHW = (blockIdx.y + 1) * tot_block_BHW / gridDim.y * HW / batch_split;

    int threadIdx_low = threadIdx.x & (BLOCK_CI_SIZE - 1);
    int threadIdx_high = threadIdx.y * (WARP_SIZE / BLOCK_CI_SIZE) | (threadIdx.x / BLOCK_CI_SIZE);

    int read_input_ci = start_block_CI + (has_hw ? threadIdx.y : threadIdx_low);
    int read_weight_ci = start_block_CI + threadIdx_low;

    #pragma unroll
    for (int i = 0; i < GROUP_CI; i++) {
        if (read_weight_ci + i * BLOCK_CI_SIZE < CI_div_G) {
            #pragma unroll
            for (int j = 0; j < BLOCK_CO_SIZE / WARP_SIZE; j++) {
                int read_weight_co = start_block_CO + threadIdx_high + j * WARP_SIZE;
                int offset = (blockIdx.z * CO_div_G + read_weight_co) * CI_div_G + read_weight_ci + i * BLOCK_CI_SIZE;
                blockW[threadIdx_low + i * BLOCK_CI_SIZE][threadIdx_high + j * WARP_SIZE] =
                    read_weight_co < CO_div_G ? weight[offset] : 0;
            }
        }
    }

    p -= 1;

    float res_grad_weight[GROUP_CI][BLOCK_CO_SIZE / WARP_SIZE];
    #pragma unroll
    for (int i = 0; i < GROUP_CI; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCK_CO_SIZE / WARP_SIZE; j++)
            res_grad_weight[i][j] = 0;
    }

    for (int k = start_block_BHW; k < end_block_BHW; k += WARP_SIZE) {
        float x[GROUP_B][GROUP_CI];
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++) {
            if (read_input_ci + i * BLOCK_CI_SIZE < CI_div_G) {
                int bhw = has_hw ? k + threadIdx.x : k + threadIdx_high;
                int b = has_hw ? bhw / HW : bhw;
                int hw = has_hw ? bhw % HW : 0;
                int offset = ((b * G + blockIdx.z) * CI_div_G + read_input_ci + i * BLOCK_CI_SIZE) * HW + hw;
                if (has_hw) x[0][i] = bhw < end_block_BHW ? input[offset] : 0;
                else blockI[threadIdx_low + i * BLOCK_CI_SIZE][threadIdx_high] = bhw < end_block_BHW ? input[offset] : 0;
            }
        }
        if (!has_hw) {
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < GROUP_CI; i++)
                x[0][i] = blockI[threadIdx.y + i * BLOCK_CI_SIZE][threadIdx.x];
        }
        #pragma unroll
        for (int s = 1; s < GROUP_B; s++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CI; i++)
                x[s][i] = __shfl_xor_sync(0xffffffff, x[0][i], WARP_SIZE / GROUP_B * s);
        }
        float res_grad_input[GROUP_B][GROUP_CI];
        #pragma unroll
        for (int s = 0; s < GROUP_B; s++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CI; i++)
                res_grad_input[s][i] = 0;
        }

//        #pragma unroll
        for (int j = 0; j < BLOCK_CO_SIZE / WARP_SIZE; j++) {
            #pragma unroll
            for (int i = 0; i < WARP_SIZE; i += BLOCK_CI_SIZE) {
                int read_output_co = start_block_CO + (has_hw ? threadIdx.y + i : threadIdx.x) + j * WARP_SIZE;
                int bhw = has_hw ? k + threadIdx.x : k + threadIdx.y + i;
                int b = has_hw ? bhw / HW : bhw;
                int hw = has_hw ? bhw % HW : 0;
                int offset = ((b * G + blockIdx.z) * CO_div_G + read_output_co) * HW + hw;
                if (bhw < end_block_BHW && read_output_co < CO_div_G) {
                    blockO[threadIdx.y + i][threadIdx.x] = __frcp_rn(output[offset]);
                    blockGO[threadIdx.y + i][threadIdx.x] = grad_output[offset];
                }
                else blockO[threadIdx.y + i][threadIdx.x] = blockGO[threadIdx.y + i][threadIdx.x] = 0;
            }
            __syncthreads();
            float res_grad_w[GROUP_CI];
            #pragma unroll
            for (int i = 0; i < GROUP_CI; i++)
                res_grad_w[i] = 0;
            #pragma unroll(4)
            for (int t = 0; t < (WARP_SIZE / GROUP_B); t++) {
                #pragma unroll
                for (int i = 0; i < GROUP_CI; i++) {
                    float sum_res;
                    #pragma unroll
                    for (int s = 0; s < GROUP_B; s++) {
                        int b = threadIdx.x ^ (WARP_SIZE / GROUP_B * s);
                        int co = threadIdx.x ^ t;
                        float w = blockW[threadIdx.y + i * BLOCK_CI_SIZE][co + j * WARP_SIZE];
                        float ro = has_hw ? blockO[co][b] : blockO[b][co];
                        float g = has_hw ? blockGO[co][b] : blockGO[b][co];
                        float res = update_backward_input<ip>(x[s][i], w, ro, g, p); // grad at b=threadIdx.x
                        if (s == 0) sum_res = res;
                        else sum_res += res;
                        if (s == GROUP_B - 1) res_grad_w[i] -= __shfl_xor_sync(0xffffffff, sum_res, t); // grad at co=threadIdx.x
                        res_grad_input[s][i] += res;
                    }

                }
            }
            #pragma unroll
            for (int i = 0; i < GROUP_CI; i++)
                res_grad_weight[i][j] += res_grad_w[i];
            __syncthreads();
        }
        #pragma unroll
        for (int s = 1; s < GROUP_B; s++) {
            #pragma unroll
            for (int i = 0; i < GROUP_CI; i++)
                res_grad_input[0][i] += __shfl_xor_sync(0xffffffff, res_grad_input[s][i], WARP_SIZE / GROUP_B * s);
        }
        if (!has_hw) {
            #pragma unroll
            for (int i = 0; i < GROUP_CI; i++)
                blockI[threadIdx.y + i * BLOCK_CI_SIZE][threadIdx.x] = res_grad_input[0][i];
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < GROUP_CI; i++)
                res_grad_input[0][i] = blockI[threadIdx_low + i * BLOCK_CI_SIZE][threadIdx_high];
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < GROUP_CI; i++) {
            if (read_input_ci + i * BLOCK_CI_SIZE < CI_div_G) {
                int bhw = has_hw ? k + threadIdx.x : k + threadIdx_high;
                int b = has_hw ? bhw / HW : bhw;
                int hw = has_hw ? bhw % HW : 0;
                if (bhw < end_block_BHW) {
                    int offset = ((b * G + blockIdx.z) * CI_div_G + read_input_ci + i * BLOCK_CI_SIZE) * HW + hw;
                    if (multiple_co_block) atomicAdd(&grad_input[offset], res_grad_input[0][i]);
                    else grad_input[offset] = res_grad_input[0][i];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < GROUP_CI; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCK_CO_SIZE / WARP_SIZE; j++)
            blockW[threadIdx.y + i * BLOCK_CI_SIZE][threadIdx.x + j * WARP_SIZE] = res_grad_weight[i][j];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < GROUP_CI; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCK_CO_SIZE / WARP_SIZE; j++) {
            float res_grad_w = blockW[threadIdx_low + i * BLOCK_CI_SIZE][threadIdx_high + j * WARP_SIZE];
            int read_weight_co = start_block_CO + threadIdx_high + j * WARP_SIZE;
            int offset = (blockIdx.z * CO_div_G + read_weight_co) * CI_div_G + read_weight_ci + i * BLOCK_CI_SIZE;
            if (read_weight_co < CO_div_G && read_weight_ci + i * BLOCK_CI_SIZE < CI_div_G)
                atomicAdd(&grad_weight[offset], res_grad_w);
        }
    }
}

#define helper_func_forward(func, dimGrid, dimBlock, ip, HW, CI_div_G, paras...) \
    const int GROUP_CI = 2; \
    const int GROUP_CO = 2; \
    const int GROUP_B = 4; \
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); \
    dim3 dimGrid((B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1, (CO / G - 1) / (BLOCK_SIZE * GROUP_CO) + 1, G); \
    if (HW == 1) { \
        if (CI_div_G % (GROUP_CI * BLOCK_SIZE) == 0) \
            func<ip, false, false, GROUP_CI, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        else func<ip, false, true, GROUP_CI, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
    } \
    else { \
        if (CI_div_G % (GROUP_CI * BLOCK_SIZE) == 0) \
            func<ip, true, false, GROUP_CI, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        else func<ip, true, true, GROUP_CI, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
    }

template <int ip>
void norm_dist<ip>::forward(const float* input, const float* weight,
                            int B, int CO, int CI, int G, int HW, float* output, float p) {
    int CI_div_G = CI / G;
    helper_func_forward(norm_dist_forward_kernel, dimGrid, dimBlock, ip, HW, CI_div_G,
                        input, weight, B, CO / G, CI_div_G, HW, G, output, p);
}

#define helper_func_forward_with_max(func, dimGrid, dimBlock, ip, has_max, HW, CI_div_G, paras...) \
    const int GROUP_CO = ip >= 0 ? 4 : 2; \
    const int GROUP_B = 4; \
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); \
    dim3 dimGrid((B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1, (CO / G - 1) / (BLOCK_SIZE * GROUP_CO) + 1, G); \
    if (HW == 1) { \
        if (CI_div_G % BLOCK_SIZE == 0) \
            func<ip, false, has_max, false, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        else func<ip, false, has_max, true, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
    } \
    else { \
        if (CI_div_G % BLOCK_SIZE == 0) \
            func<ip, true, has_max, false, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        else func<ip, true, has_max, true, GROUP_CO, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
    }

template <int ip>
void norm_dist<ip>::forward_with_max(const float* input, const float* weight, const float* max_output,
                                     int B, int CO, int CI, int G, int HW, float* output, float p) {
    int CI_div_G = CI / G;
    if (max_output) {
        helper_func_forward_with_max(norm_dist_forward_kernel, dimGrid, dimBlock, ip, true, HW, CI_div_G,
                                     input, weight, max_output, B, CO / G, CI_div_G, HW, G, output, p);
    }
    else {
        helper_func_forward_with_max(norm_dist_forward_kernel, dimGrid, dimBlock, ip, false, HW, CI_div_G,
                                     input, weight, max_output, B, CO / G, CI_div_G, HW, G, output, p);
    }
}

#define helper_func_backward_input(func, dimGrid, dimBlock, ip, HW, CO_div_G, paras...) \
    const int GROUP_CI = 4; \
    const int GROUP_B = 4; \
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); \
    dim3 dimGrid((B * HW - 1) / (BLOCK_SIZE * GROUP_B) + 1, (CI / G - 1) / (BLOCK_SIZE * GROUP_CI) + 1, G); \
    int CO_div_G = CO / G; \
    if (HW == 1) { \
        if (CO_div_G % BLOCK_SIZE == 0) \
            func<ip, false, false, GROUP_CI, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        else func<ip, false, true, GROUP_CI, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
    } \
    else { \
        if (CO_div_G % BLOCK_SIZE == 0) \
            func<ip, true, false, GROUP_CI, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
        else func<ip, true, true, GROUP_CI, GROUP_B><<<dimGrid, dimBlock>>>(paras); \
    }

template <int ip>
void norm_dist<ip>::backward_input(const float* grad_output, const float* input, const float* weight, const float* output,
                                   int B, int CO, int CI, int G, int HW, float* grad_input, float p) {
    helper_func_backward_input(norm_dist_backward_input_kernel, dimGrid, dimBlock, ip, HW, CO_div_G,
                               grad_output, input, weight, output, B, CO_div_G, CI / G, HW, G, grad_input, p);
}

#define helper_func_backward_input_weight(func, dimGrid, dimBlock, ip, CO_div_G, HW, paras...) \
    const int GROUP_CI_CONV = 4; \
    const int GROUP_CI_FC = 4; \
    const int GROUP_B = 2; \
    cudaDeviceProp deviceProp; \
    cudaGetDeviceProperties(&deviceProp, 0); \
    dim3 dimBlock(WARP_SIZE, BLOCK_CI_SIZE); \
    int num_block_ci = (CI / G - 1) / (BLOCK_CI_SIZE * (HW == 1 ? GROUP_CI_FC : GROUP_CI_CONV)) + 1; \
    int num_block_co = (CO_div_G - 1) / MAX_BLOCK_CO_SIZE + 1; \
    if (HW > 1 && CO_div_G > 2 * WARP_SIZE && CO_div_G <= 3 * WARP_SIZE) num_block_co = 1; \
    int num_block_b1 = max(deviceProp.multiProcessorCount * 36 / (num_block_ci * num_block_co * G), 1); \
    int num_block_b = min(num_block_b1, B * min(HW, BATCH_SPLIT)); \
    dim3 dimGrid(num_block_ci * num_block_co, num_block_b, G); \
    cudaMemset(grad_weight, 0, CO * (CI / G) * sizeof(float)); \
    if (HW == 1) { \
        cudaMemset(grad_input, 0, B * CI * HW * sizeof(float)); \
        func<ip, false, true, GROUP_CI_FC, GROUP_B, MAX_BLOCK_CO_SIZE><<<dimGrid, dimBlock>>>(paras); \
    } \
    else if (CO_div_G <= WARP_SIZE) func<ip, true, false, GROUP_CI_CONV, GROUP_B, WARP_SIZE><<<dimGrid, dimBlock>>>(paras); \
    else if (CO_div_G <= 2 * WARP_SIZE) func<ip, true, false, GROUP_CI_CONV, GROUP_B, 2 * WARP_SIZE><<<dimGrid, dimBlock>>>(paras); \
    else if (CO_div_G <= 3 * WARP_SIZE) func<ip, true, false, GROUP_CI_CONV, GROUP_B, 3 * WARP_SIZE><<<dimGrid, dimBlock>>>(paras); \
    else { \
        cudaMemset(grad_input, 0, B * CI * HW * sizeof(float)); \
        func<ip, true, true, GROUP_CI_CONV, GROUP_B, MAX_BLOCK_CO_SIZE><<<dimGrid, dimBlock>>>(paras); \
    }

template <int ip>
void norm_dist<ip>::backward_input_weight(const float* grad_output, const float* input, const float* weight,
                                          const float* output, int B, int CO, int CI, int G, int HW,
                                          float* grad_input, float* grad_weight, float p) {
    int CO_div_G = CO / G;
    helper_func_backward_input_weight(norm_dist_backward_input_weight_kernel, dimGrid, dimBlock, ip, CO_div_G, HW,
                grad_output, input, weight, output, B, CO / G, CI / G, HW, G, grad_input, grad_weight, p, num_block_ci);
}

#define build_p(ip) \
    template struct norm_dist<ip>;

build_p(0)
build_p(1)
build_p(2)
build_p(3)
build_p(4)
build_p(5)
build_p(6)
build_p(7)
build_p(8)
build_p(-1)
