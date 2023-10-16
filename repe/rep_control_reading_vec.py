# wrapping classes
import torch
import numpy as np

class WrappedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.normalize = False

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        if isinstance(output, tuple):
            self.output = output[0]
            modified = output[0]
        else:
            self.output = output
            modified = output

            
        if self.controller is not None:
        
            norm_pre = torch.norm(modified, dim=-1, keepdim=True)

            if self.mask is not None:
                mask = self.mask

            # we should ignore the padding tokens when doing the activation addition
            # mask has ones for non padding tokens and zeros at padding tokens.
            # only tested this on left padding
            elif "position_ids" in kwargs:
                pos = kwargs["position_ids"]
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                target_shape = modified.shape
                mask = (col_indices >= zero_indices).float().reshape(target_shape[0], target_shape[1], 1)
                mask = mask.to(modified.dtype)
            else:
                # print(f"Warning: block {self.block_name} does not contain information 'position_ids' about token types. When using batches this can lead to unexpected results.")
                mask = 1.0

            if len(self.controller.shape) == 1:
                self.controller = self.controller.reshape(1, 1, -1)
            assert len(self.controller.shape) == len(modified.shape), f"Shape of controller {self.controller.shape} does not match shape of modified {modified.shape}."

            self.controller = self.controller.to(modified.device)
            if type(mask) == torch.Tensor:
                mask = mask.to(modified.device)
            if isinstance(self.token_pos, int):
                modified[:, self.token_pos] = modified[:, self.token_pos] + self.controller * mask
            elif isinstance(self.token_pos, list) or isinstance(self.token_pos, tuple) or isinstance(self.token_pos, np.ndarray):
                modified[:, self.token_pos] = modified[:, self.token_pos] + self.controller * mask
            elif isinstance(self.token_pos, str):
                if self.token_pos == "end":
                    len_token = self.controller.shape[1]
                    modified[:, -len_token:] = modified[:, -len_token:] + self.controller * mask
                elif self.token_pos == "start":
                    len_token = self.controller.shape[1]
                    modified[:, :len_token] = modified[:, :len_token] + self.controller * mask
                else:
                    assert False, f"Unknown token position {self.token_pos}."
            else:
                modified = modified + self.controller * mask

            if self.normalize:
                norm_post = torch.norm(modified, dim=-1, keepdim=True)
                modified = modified / norm_post * norm_pre
            
        if isinstance(output, tuple):
            output = (modified,) + output[1:] 
        else:
            output = modified
        
        return output

    def set_controller(self, activations, token_pos=None, masks=None, normalize=False):
        self.normalize = normalize
        self.controller = activations.squeeze()
        self.mask = masks
        self.token_pos = token_pos
        
    def reset(self):
        self.output = None
        self.controller = None
        self.mask = None

    def set_masks(self, masks):
        self.mask = masks

    
class WrappedReadingVecModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def generate(self, prompt, max_new_tokens=100, random_seed=0, use_cache=True):
        with torch.no_grad():
            torch.random.manual_seed(random_seed)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            attention_mask = inputs.attention_mask.to(self.model.device)
            generate_ids = self.model.generate(inputs.input_ids.to(self.model.device), attention_mask=attention_mask, max_new_tokens=max_new_tokens, use_cache=use_cache)
            return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.model.device)).logits
            return logits
        
    def run_prompt(self, prompt, **kwargs):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            return output
        
    def wrap_self_attn(self, layer_id):
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = self.model.model.layers[layer_id].block.self_attn
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.self_attn = WrappedBlock(block)
        else:
            block = self.model.model.layers[layer_id].self_attn
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].self_attn = WrappedBlock(block)
    
    def wrap_mlp(self, layer_id):
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = self.model.model.layers[layer_id].block.mlp
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.mlp = WrappedBlock(block)
        else:
            block = self.model.model.layers[layer_id].mlp
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].mlp = WrappedBlock(block)
        
    def wrap_input_layernorm(self, layer_id):
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = self.model.model.layers[layer_id].block.input_layernorm
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.input_layernorm = WrappedBlock(block)
        else:
            block = self.model.model.layers[layer_id].input_layernorm
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].input_layernorm = WrappedBlock(block)
        
    def wrap_post_attention_layernorm(self, layer_id):
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = self.model.model.layers[layer_id].block.post_attention_layernorm
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].block.post_attention_layernorm = WrappedBlock(block)
        else:
            block = self.model.model.layers[layer_id].post_attention_layernorm
            if not self.is_wrapped(block):
                self.model.model.layers[layer_id].post_attention_layernorm = WrappedBlock(block)
        
    def wrap_decoder_block(self, layer_id):
        block = self.model.model.layers[layer_id]
        if not self.is_wrapped(block):
            self.model.model.layers[layer_id] = WrappedBlock(block)
        
    
    def wrap_all(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            self.wrap_self_attn(layer_id)
            self.wrap_mlp(layer_id)
            self.wrap_input_layernorm(layer_id)
            self.wrap_post_attention_layernorm(layer_id)
            self.wrap_decoder_block(layer_id)
            
    def wrap_block(self, layer_ids, block_name):
        def _wrap_block(layer_id, block_name):
            if block_name == 'self_attn':
                self.wrap_self_attn(layer_id)
            elif block_name == 'mlp':
                self.wrap_mlp(layer_id)
            elif block_name == 'input_layernorm':
                self.wrap_input_layernorm(layer_id)
            elif block_name == 'post_attention_layernorm':
                self.wrap_post_attention_layernorm(layer_id)
            elif block_name == 'decoder_block':
                self.wrap_decoder_block(layer_id)
            else:
                assert False, f"No block named {block_name}."

        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            for layer_id in layer_ids:
                _wrap_block(layer_id, block_name)
        else:
            _wrap_block(layer_ids, block_name)

            
    def get_activations(self, layer_ids, block_name='decoder_block'):

        def _get_activations(layer_id, block_name):
            current_layer = self.model.model.layers[layer_id]

            if self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name == 'decoder_block':
                    return current_layer.output
                elif block_name == 'self_attn' and self.is_wrapped(current_block.self_attn):
                    return current_block.self_attn.output
                elif block_name == 'mlp' and self.is_wrapped(current_block.mlp):
                    return current_block.mlp.output
                elif block_name == 'input_layernorm' and self.is_wrapped(current_block.input_layernorm):
                    return current_block.input_layernorm.output
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_block.post_attention_layernorm):
                    return current_block.post_attention_layernorm.output
                else:
                    assert False, f"No wrapped block named {block_name}."

            else:
                if block_name == 'self_attn' and self.is_wrapped(current_layer.self_attn):
                    return current_layer.self_attn.output
                elif block_name == 'mlp' and self.is_wrapped(current_layer.mlp):
                    return current_layer.mlp.output
                elif block_name == 'input_layernorm' and self.is_wrapped(current_layer.input_layernorm):
                    return current_layer.input_layernorm.output
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_layer.post_attention_layernorm):
                    return current_layer.post_attention_layernorm.output
                else:
                    assert False, f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            activations = {}
            for layer_id in layer_ids:
                activations[layer_id] = _get_activations(layer_id, block_name)
            return activations
        else:
            return _get_activations(layer_ids, block_name)


    def set_controller(self, layer_ids, activations, block_name='decoder_block', token_pos=None, masks=None, normalize=False):

        def _set_controller(layer_id, activations, block_name, masks, normalize):
            current_layer = self.model.model.layers[layer_id]

            if block_name == 'decoder_block':
                current_layer.set_controller(activations, token_pos, masks, normalize)
            elif self.is_wrapped(current_layer):
                current_block = current_layer.block  
                if block_name == 'self_attn' and self.is_wrapped(current_block.self_attn):
                    current_block.self_attn.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'mlp' and self.is_wrapped(current_block.mlp):
                    current_block.mlp.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'input_layernorm' and self.is_wrapped(current_block.input_layernorm):
                    current_block.input_layernorm.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_block.post_attention_layernorm):
                    current_block.post_attention_layernorm.set_controller(activations, token_pos, masks, normalize)
                else:
                    return f"No wrapped block named {block_name}."

            else:
                if block_name == 'self_attn' and self.is_wrapped(current_layer.self_attn):
                    current_layer.self_attn.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'mlp' and self.is_wrapped(current_layer.mlp):
                    current_layer.mlp.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'input_layernorm' and self.is_wrapped(current_layer.input_layernorm):
                    current_layer.input_layernorm.set_controller(activations, token_pos, masks, normalize)
                elif block_name == 'post_attention_layernorm' and self.is_wrapped(current_layer.post_attention_layernorm):
                    current_layer.post_attention_layernorm.set_controller(activations, token_pos, masks, normalize)
                else:
                    return f"No wrapped block named {block_name}."
                
        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            assert isinstance(activations, dict), "activations should be a dictionary"
            for layer_id in layer_ids:
                _set_controller(layer_id, activations[layer_id], block_name, masks, normalize)
        else:
            _set_controller(layer_ids, activations, block_name, masks, normalize)
      
        
    def reset(self):
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.reset()
                if self.is_wrapped(layer.block.self_attn):
                    layer.block.self_attn.reset()
                if self.is_wrapped(layer.block.mlp):
                    layer.block.mlp.reset()
                if self.is_wrapped(layer.block.input_layernorm):
                    layer.block.input_layernorm.reset()
                if self.is_wrapped(layer.block.post_attention_layernorm):
                    layer.block.post_attention_layernorm.reset()
            else:   
                if self.is_wrapped(layer.self_attn):
                    layer.self_attn.reset()
                if self.is_wrapped(layer.mlp):
                    layer.mlp.reset()
                if self.is_wrapped(layer.input_layernorm):
                    layer.input_layernorm.reset()
                if self.is_wrapped(layer.post_attention_layernorm):
                    layer.post_attention_layernorm.reset()

    def set_masks(self, masks):
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.set_masks(masks)
                if self.is_wrapped(layer.block.self_attn):
                    layer.block.self_attn.set_masks(masks)
                if self.is_wrapped(layer.block.mlp):
                    layer.block.mlp.set_masks(masks)
                if self.is_wrapped(layer.block.input_layernorm):
                    layer.block.input_layernorm.set_masks(masks)
                if self.is_wrapped(layer.block.post_attention_layernorm):
                    layer.block.post_attention_layernorm.set_masks(masks)
            else:   
                if self.is_wrapped(layer.self_attn):
                    layer.self_attn.set_masks(masks)
                if self.is_wrapped(layer.mlp):
                    layer.mlp.set_masks(masks)
                if self.is_wrapped(layer.input_layernorm):
                    layer.input_layernorm.set_masks(masks)
                if self.is_wrapped(layer.post_attention_layernorm):
                    layer.post_attention_layernorm.set_masks(masks)


    def is_wrapped(self, block):
        if hasattr(block, 'block'):
            return True
        return False
    
    def unwrap(self):
        for l, layer in enumerate(self.model.model.layers):
            if self.is_wrapped(layer):
                self.model.model.layers[l] = layer.block
            if self.is_wrapped(self.model.model.layers[l].self_attn):
                self.model.model.layers[l].self_attn = self.model.model.layers[l].self_attn.block
            if self.is_wrapped(self.model.model.layers[l].mlp):
                self.model.model.layers[l].mlp = self.model.model.layers[l].mlp.block
            if self.is_wrapped(self.model.model.layers[l].input_layernorm):
                self.model.model.layers[l].input_layernorm = self.model.model.layers[l].input_layernorm.block
            if self.is_wrapped(self.model.model.layers[l].post_attention_layernorm):
                self.model.model.layers[l].post_attention_layernorm = self.model.model.layers[l].post_attention_layernorm.block
