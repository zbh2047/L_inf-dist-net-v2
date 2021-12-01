#include <torch/extension.h>
#include "norm_dist.h"

#define MIN_NORMALIZED_P 10.0

typedef torch::Tensor Tensor;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define GET_PTR(tensor) (tensor).data_ptr<float>()

#define call_intp(type, fun, p, paras...) {\
    type<p>::fun(paras); \
}
#define call_p(type, fun, p, paras...) { \
    static_assert(MIN_INSTANTIATED_P == 0); \
    static_assert(MAX_INSTANTIATED_P == 8); \
    if ((p) == float(int(p))) { \
        switch (int(p)) { \
            case 0: call_intp(type, fun, 0, paras); break; \
            case 1: call_intp(type, fun, 1, paras); break; \
            case 2: call_intp(type, fun, 2, paras); break; \
            case 3: call_intp(type, fun, 3, paras); break; \
            case 4: call_intp(type, fun, 4, paras); break; \
            case 5: call_intp(type, fun, 5, paras); break; \
            case 6: call_intp(type, fun, 6, paras); break; \
            case 7: call_intp(type, fun, 7, paras); break; \
            case 8: call_intp(type, fun, 8, paras); break; \
            default: call_intp(type, fun, -1, paras); \
        } \
    } \
    else call_intp(type, fun, -1, paras); \
}
#define call(type, fun, paras...) { \
    type::fun(paras); \
}

void norm_dist_forward(Tensor& input, Tensor& weight, Tensor& output, int G, float p) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    if (p <= MIN_NORMALIZED_P) {
        call_p(norm_dist, forward_with_max, p,
               GET_PTR(input), GET_PTR(weight), nullptr, B, CO, CI, G, HW, GET_PTR(output), p);
    }
    else {
        call_p(norm_dist, forward, p,
               GET_PTR(input), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(output), p);
    }
}

void norm_dist_forward_with_max(Tensor& input, Tensor& weight, Tensor& max_output, Tensor& output,
                                int G, float p) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(max_output);
    CHECK_INPUT(output);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call_p(norm_dist, forward_with_max, p,
           GET_PTR(input), GET_PTR(weight), GET_PTR(max_output), B, CO, CI, G, HW, GET_PTR(output), p);
}

void norm_dist_backward_input(Tensor& grad_output, Tensor& input, Tensor& weight, Tensor& output,
                              Tensor& grad_input, int G, float p) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(grad_input);
    int B = grad_output.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call_p(norm_dist, backward_input, p - 1,
           GET_PTR(grad_output), GET_PTR(input), GET_PTR(weight), GET_PTR(output), B, CO, CI, G, HW,
           GET_PTR(grad_input), p);
}

void norm_dist_backward_input_weight(Tensor& grad_output, Tensor& input, Tensor& weight, Tensor& output,
                                     Tensor& grad_input, Tensor& grad_weight, int G, float p) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(grad_input);
    CHECK_INPUT(grad_weight);
    int B = grad_output.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call_p(norm_dist, backward_input_weight, p - 1,
           GET_PTR(grad_output), GET_PTR(input), GET_PTR(weight), GET_PTR(output), B, CO, CI, G, HW,
           GET_PTR(grad_input), GET_PTR(grad_weight), p);
}

void inf_dist_forward(Tensor& input, Tensor& weight, Tensor& output, Tensor& pos, int G) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    CHECK_INPUT(pos);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call(inf_dist, forward,
         GET_PTR(input), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(output), pos.data_ptr<int>());
}

void inf_dist_forward_nograd(Tensor& input, Tensor& weight, Tensor& output, int G) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    int B = input.size(0), CO = output.size(1), CI = input.size(1), HW = input.size(2);
    call(inf_dist, forward_nograd,
         GET_PTR(input), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(output));
}

void inf_dist_backward_input(Tensor& grad_output, Tensor& pos, Tensor& grad_input, int G) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(pos);
    CHECK_INPUT(grad_input);
    int B = grad_output.size(0), CO = grad_output.size(1), CI = grad_input.size(1), HW = grad_input.size(2);
    call(inf_dist, backward_input,
         GET_PTR(grad_output), pos.data_ptr<int>(), B, CO, CI, G, HW, GET_PTR(grad_input));
}

void inf_dist_backward_input_weight(Tensor& grad_output, Tensor& pos, Tensor& grad_input, Tensor& grad_weight,
                                    int G) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(pos);
    CHECK_INPUT(grad_input);
    CHECK_INPUT(grad_weight);
    int B = grad_output.size(0), CO = grad_output.size(1), CI = grad_weight.size(1) * G, HW = grad_output.size(2);
    call(inf_dist, backward_input_weight,
         GET_PTR(grad_output), pos.data_ptr<int>(), B, CO, CI, G, HW, GET_PTR(grad_input), GET_PTR(grad_weight));
}



void bound_inf_dist_forward_nograd(Tensor& inputL, Tensor& inputU, Tensor& weight, Tensor& outputL, Tensor& outputU,
                                   int G) {
    CHECK_INPUT(inputL);
    CHECK_INPUT(inputU);
    CHECK_INPUT(weight);
    CHECK_INPUT(outputL);
    CHECK_INPUT(outputU);
    int B = inputL.size(0), CO = outputL.size(1), CI = inputL.size(1), HW = inputL.size(2);
    call(bound_inf_dist, forward_nograd,
         GET_PTR(inputL), GET_PTR(inputU), GET_PTR(weight), B, CO, CI, G, HW, GET_PTR(outputL), GET_PTR(outputU));
}

void bound_inf_dist_forward(Tensor& inputL, Tensor& inputU, Tensor& weight, Tensor& outputL, Tensor& outputU,
                            Tensor& posL, Tensor& posU, int G) {
    CHECK_INPUT(inputL);
    CHECK_INPUT(inputU);
    CHECK_INPUT(weight);
    CHECK_INPUT(outputL);
    CHECK_INPUT(outputU);
    CHECK_INPUT(posL);
    CHECK_INPUT(posU);
    int B = inputL.size(0), CO = outputL.size(1), CI = inputL.size(1), HW = inputL.size(2);
    call(bound_inf_dist, forward,
         GET_PTR(inputL), GET_PTR(inputU), GET_PTR(weight), B, CO, CI, G, HW,
         GET_PTR(outputL), GET_PTR(outputU), posL.data_ptr<int>(), posU.data_ptr<int>());
}

void bound_inf_dist_backward_input(Tensor& grad_outputL, Tensor& grad_outputU, Tensor& posL, Tensor& posU,
                                   Tensor& grad_inputL, Tensor& grad_inputU, int G) {
    CHECK_INPUT(grad_outputL);
    CHECK_INPUT(grad_outputU);
    CHECK_INPUT(posL);
    CHECK_INPUT(posU);
    CHECK_INPUT(grad_inputL);
    CHECK_INPUT(grad_inputU);
    int B = grad_outputL.size(0), CO = grad_outputL.size(1), CI = grad_inputL.size(1), HW = grad_inputL.size(2);
    call(bound_inf_dist, backward_input,
         GET_PTR(grad_outputL), GET_PTR(grad_outputU), posL.data_ptr<int>(), posU.data_ptr<int>(),
         B, CO, CI, G, HW, GET_PTR(grad_inputL), GET_PTR(grad_inputU));
}

void bound_inf_dist_backward_input_weight(Tensor& grad_outputL, Tensor& grad_outputU, Tensor& posL, Tensor& posU,
                                          Tensor& grad_inputL, Tensor& grad_inputU, Tensor& grad_weight,
                                          int G) {
    CHECK_INPUT(grad_outputL);
    CHECK_INPUT(grad_outputU);
    CHECK_INPUT(posL);
    CHECK_INPUT(posU);
    CHECK_INPUT(grad_inputL);
    CHECK_INPUT(grad_inputU);
    CHECK_INPUT(grad_weight);
    int B = grad_outputL.size(0), CO = grad_outputL.size(1), CI = grad_weight.size(1) * G, HW = grad_outputL.size(2);
    call(bound_inf_dist, backward_input_weight,
         GET_PTR(grad_outputL), GET_PTR(grad_outputU), posL.data_ptr<int>(), posU.data_ptr<int>(),
         B, CO, CI, G, HW, GET_PTR(grad_inputL), GET_PTR(grad_inputU), GET_PTR(grad_weight));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("norm_dist_forward", &norm_dist_forward);
    m.def("norm_dist_forward_with_max", &norm_dist_forward_with_max);
    m.def("norm_dist_backward_input", &norm_dist_backward_input);
    m.def("norm_dist_backward_input_weight", &norm_dist_backward_input_weight);
    m.def("inf_dist_forward", &inf_dist_forward);
    m.def("inf_dist_forward_nograd", &inf_dist_forward_nograd);
    m.def("inf_dist_backward_input", &inf_dist_backward_input);
    m.def("inf_dist_backward_input_weight", &inf_dist_backward_input_weight);
    m.def("bound_inf_dist_forward", &bound_inf_dist_forward);
    m.def("bound_inf_dist_forward_nograd", &bound_inf_dist_forward_nograd);
    m.def("bound_inf_dist_backward_input", &bound_inf_dist_backward_input);
    m.def("bound_inf_dist_backward_input_weight", &bound_inf_dist_backward_input_weight);
}