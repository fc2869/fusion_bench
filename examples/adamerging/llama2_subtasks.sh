# Layer-wise adamerging
# module load gcc cuda;
# module load python3;
export HF_HOME=/work/10269/fcyin/.cache;
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 fusion_bench \
    method=adamerging/llama2_subtasks \
    modelpool=CausalLMPool/llama2_for_causallm \
    taskpool=pen_pen\
    # fabric=llama_ddp

