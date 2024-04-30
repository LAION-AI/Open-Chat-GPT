import transformers


def freeze_top_n_layers(model, target_layers):
    # it's possible we can simply detect which module is a ModuleList
    # and simply freeze the module without doing string parsing
    for name, param in model.named_parameters():
        if "embed" in name:
            param.requires_grad = False
        elif ".layer" in name or ".h." in name:
            tokens = name.split(".")
            layer_ = None
            for token in tokens:
                if token.isdigit():
                    layer_ = int(token)
                    break
            if layer_ is not None and layer_ < target_layers:
                # print('freeze ', layer_, name)
                param.requires_grad = False
    return model


def get_specific_model(
    model_name,
    seq2seqmodel=False,
    without_head=False,
    cache_dir=".cache",
    quantization=False,
    **kwargs,
):
    if without_head:
        model = transformers.AutoModel.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    elif seq2seqmodel:
        # encoder-decoder support for Flan-T5 like models
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    else:
        if "falcon-7b" in model_name:
            # temporary hack until tiiuae/falcon-7b uses the transformer's Falcon impl by default
            # in-library PR was reverted https://huggingface.co/tiiuae/falcon-7b/commit/378337427557d1df3e742264a2901a49f25d4eb1
            model = transformers.models.falcon.modeling_falcon.FalconForCausalLM.from_pretrained(
                model_name, cache_dir=cache_dir, **kwargs
            )
        else:
            if "falcon" in model_name:
                kwargs["trust_remote_code"] = True
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    return model
