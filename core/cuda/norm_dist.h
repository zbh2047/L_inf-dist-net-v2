#define MIN_INSTANTIATED_P 0
#define MAX_INSTANTIATED_P 8

template <int ip> struct norm_dist {
    static void forward(const float* input, const float* weight,
                        int B, int CO, int CI, int G, int HW, float* output, float p);
    static void forward_with_max(const float* input, const float* weight, const float* max_output,
                                 int B, int CO, int CI, int G, int HW, float* output, float p);
    static void backward_input(const float* grad_output, const float* input, const float* weight, const float* output,
                               int B, int CO, int CI, int G, int HW, float* grad_input, float p);
    static void backward_input_weight(const float* grad_output, const float* input, const float* weight,
                                      const float* output, int B, int CO, int CI, int G, int HW,
                                      float* grad_input, float* grad_weight, float p);
};

struct inf_dist {
    static void forward_nograd(const float* input, const float* weight,
                               int B, int CO, int CI, int G, int HW, float* output);
    static void forward(const float* input, const float* weight,
                        int B, int CO, int CI, int G, int HW, float* output, int* pos);
    static void backward_input(const float* grad_output, const int* pos,
                               int B, int CO, int CI, int G, int HW, float* grad_input);
    static void backward_input_weight(const float* grad_output, const int* pos,
                                      int B, int CO, int CI, int G, int HW, float* grad_input, float* grad_weight);
};

struct bound_inf_dist {
    static void forward_nograd(const float* inputL, const float* inputU, const float* weight,
                                        int B, int CO, int CI, int G, int HW, float* outputL, float* outputU);
    static void forward(const float* inputL, const float* inputU, const float* weight,
                        int B, int CO, int CI, int G, int HW, float* outputL, float* outputU, int* posL, int* posU);
    static void backward_input(const float* grad_outputL, const float* grad_outputU,
                               const int* posL, const int* posU, int B, int CO, int CI, int G, int HW,
                               float* grad_inputL, float* grad_inputU);
    static void backward_input_weight(const float* grad_outputL, const float* grad_outputU,
                                      const int* posL, const int* posU, int B, int CO, int CI, int G, int HW,
                                      float* grad_inputL, float* grad_inputU, float* grad_weight);
};
