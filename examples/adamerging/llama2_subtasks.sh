# Layer-wise adamerging
fusion_bench \
    method=adamerging/llama2_subtasks \
    method.optimizer.lr=1e-3 \
    modelpool=CausalLMPool/llama2_for_causallm \
    taskpool=pen_pen
