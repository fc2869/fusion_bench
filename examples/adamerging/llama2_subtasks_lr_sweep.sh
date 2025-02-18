# Layer-wise adamerging
export HF_HOME=/work/10269/fcyin/.cache;
# for lr in 5e-3 1e-3 5e-4 1e-4; do
for lr in 5e-3; do
    export lr;
    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 fusion_bench \
        method=adamerging/llama2_subtasks \
        method.init_values=0.5 \
        method.save_interval=300 \
        method.lr=$lr \
        method.max_steps=900 \
        modelpool=CausalLMPool/llama2_for_causallm \
        taskpool=skillmix_literary_rhetorical\
        # fabric=llama_ddp
done

