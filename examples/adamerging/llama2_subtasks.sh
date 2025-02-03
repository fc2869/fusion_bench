# Layer-wise adamerging
HYDRA_FULL_ERROR=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 fusion_bench \
    method=adamerging/llama2_subtasks \
    modelpool=CausalLMPool/llama2_for_causallm \
    taskpool=pen_pen\
    # fabric=llama_ddp

