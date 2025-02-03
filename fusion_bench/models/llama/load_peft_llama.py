from transformers import LlamaForCausalLM
from peft import PeftModel
def load_llama_model_hf_with_pad(
    base_model_name: str,
    model_kwargs: dict,
):
    model = LlamaForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model.resize_token_embeddings(model.config.vocab_size + 1)
    return model

def load_lora_llama_model_hf(
    base_model_name: str,
    peft_name: str,
    merge_and_unload: bool = False,
):
    """
    Load a LoRA (Low-Rank Adaptation) vision model from Hugging Face.

    This function loads a vision model and applies a LoRA adaptation to it. The model can be optionally merged and unloaded.

    Parameters:
        base_model_name (str): The name of the base vision model to load from Hugging Face.
        peft_name (str): The name of the LoRA adaptation to apply to the base model.
        merge_and_unload (bool, optional): If True, the LoRA adaptation is merged into the base model and the LoRA layers are removed. Defaults to False.
        return_vison_model (bool, optional): If False, the full CLIPVisionModel is returned. If True, only the vision model (`CLIPVisionTransformer`) is returned. Defaults to True.

    Returns:
        PeftModel: The adapted vision model, optionally merged and unloaded.
    """
    model = LlamaForCausalLM.from_pretrained(base_model_name)

    peft_model = PeftModel.from_pretrained(model, peft_name, is_trainable=True)
    if merge_and_unload:
        model = peft_model.merge_and_unload()
    else:
        model = peft_model

    return model