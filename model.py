"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from hyper_conn import hyper_conn_init_func

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, init_hc, expand_stream, reduce_stream, layer_index=None):
        super().__init__()
        self.init_hc = init_hc
        self.expand_stream = expand_stream
        self.reduce_stream = reduce_stream
        self.layer_index = layer_index

        self.branch_attn = nn.Sequential(
            LayerNorm(config.n_embd, bias=config.bias), 
            CausalSelfAttention(config)
        )
        self.branch_mlp  = nn.Sequential(
            LayerNorm(config.n_embd, bias=config.bias), 
            MLP(config)
        )

        if self.init_hc is not None:
            self.hc_attn = self.init_hc(
                dim = config.n_embd,
                branch = self.branch_attn,
            )
            self.hc_mlp = self.init_hc(
                dim = config.n_embd,
                branch = self.branch_mlp,
            )

    def forward(self, x):
        exp_s = self.expand_stream
        rdc_s = self.reduce_stream
        
        if self.init_hc is not None:
            assert exp_s is not None and rdc_s is not None
            x = self.hc_attn(x)
            x = self.hc_mlp(x)
        else:
            x = x + self.branch_attn(x)
            x = x + self.branch_mlp(x)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    hyper_conn_n: int = 1
    hyper_conn_type: str = "none"
    atbp_permutations: list = None  # For atbp_mhc: list of permutation specs, e.g. ["direct", "reverse"]; None = default ["direct","reverse"]
    alsb_permutations: list = None  # Legacy alias for atbp_permutations
    astbp_permutations: list = None  # For astbp_mhc: same format as atbp_permutations
    amstbp_permutations: list = None  # For amstbp_mhc: same format as atbp_permutations
    altbp_permutations: list = None  # For altbp_mhc: same format as atbp_permutations
    lmamstbp_lambda_init: float | None = None  # For lmamstbp_mhc: init for λ (default 0.05)
    lmamstbp_mu_init: float | None = None  # For lmamstbp_mhc: init for μ (default 0.01)
    lmaltbp_lambda_init: float | None = None  # For lmaltbp_mhc: init for λ (default 0.05)
    lmaltbp_mu_init: float | None = None  # For lmaltbp_mhc: init for μ (default 0.01)
    ortbp_log_stats: bool = False  # For ortbp2n_mhc: log residual-chart diagnostics

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        init_hc, expand_stream, reduce_stream = hyper_conn_init_func(
            config.hyper_conn_type,
            config.hyper_conn_n,
            atbp_permutations=getattr(config, 'atbp_permutations', None),
            alsb_permutations=getattr(config, 'alsb_permutations', None),
            astbp_permutations=getattr(config, 'astbp_permutations', None),
            amstbp_permutations=getattr(config, 'amstbp_permutations', None),
            altbp_permutations=getattr(config, 'altbp_permutations', None),
            lmamstbp_lambda_init=getattr(config, 'lmamstbp_lambda_init', None),
            lmamstbp_mu_init=getattr(config, 'lmamstbp_mu_init', None),
            lmaltbp_lambda_init=getattr(config, 'lmaltbp_lambda_init', None),
            lmaltbp_mu_init=getattr(config, 'lmaltbp_mu_init', None),
            ortbp_log_stats=getattr(config, 'ortbp_log_stats', False),
        )
        self.expand_stream = expand_stream
        self.reduce_stream = reduce_stream

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([
                Block(
                    config, 
                    init_hc, 
                    expand_stream, 
                    reduce_stream, 
                    layer_index=layer_index,
                ) for layer_index in range(config.n_layer)
            ]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        if self.expand_stream is not None:
            x = self.expand_stream(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if self.reduce_stream is not None:
            x = self.reduce_stream(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if loss_mask is not None:
                # Masked loss: only compute loss on masked positions
                loss_per_token = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1), 
                    reduction='none'
                )
                mask_flat = loss_mask.view(-1)
                loss = (loss_per_token * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        device_type,
        module_param_group_overrides=None,
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # allow modules to explicitly opt specific parameters out of weight decay,
        # even when they are 2D tensors that would normally land in the decay group
        explicit_nodecay_names = set()
        for module_name, module in self.named_modules():
            if not hasattr(module, "no_weight_decay_param_names"):
                continue
            local_names = module.no_weight_decay_param_names()
            if not local_names:
                continue
            prefix = f"{module_name}." if module_name else ""
            explicit_nodecay_names.update(f"{prefix}{name}" for name in local_names)
        explicit_group_assignments = {}
        if module_param_group_overrides:
            for module_name, module in self.named_modules():
                if not hasattr(module, "optimizer_param_group_names"):
                    continue
                local_groups = module.optimizer_param_group_names()
                if not local_groups:
                    continue
                prefix = f"{module_name}." if module_name else ""
                for group_name, local_param_names in local_groups.items():
                    if group_name not in module_param_group_overrides:
                        continue
                    for local_name in local_param_names:
                        full_name = f"{prefix}{local_name}"
                        if full_name not in param_dict:
                            continue
                        prev_group_name = explicit_group_assignments.get(full_name)
                        if prev_group_name is not None and prev_group_name != group_name:
                            raise ValueError(
                                f"Parameter {full_name} was assigned to both "
                                f"{prev_group_name} and {group_name}"
                            )
                        explicit_group_assignments[full_name] = group_name
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [
            p for n, p in param_dict.items()
            if p.dim() >= 2 and n not in explicit_nodecay_names and n not in explicit_group_assignments
        ]
        nodecay_params = [
            p for n, p in param_dict.items()
            if (p.dim() < 2 or n in explicit_nodecay_names) and n not in explicit_group_assignments
        ]
        optim_groups = []
        if decay_params:
            optim_groups.append({
                'params': decay_params,
                'weight_decay': weight_decay,
                'lr': learning_rate,
                'betas': betas,
                'group_name': 'default_decay',
            })
        if nodecay_params:
            optim_groups.append({
                'params': nodecay_params,
                'weight_decay': 0.0,
                'lr': learning_rate,
                'betas': betas,
                'group_name': 'default_nodecay',
            })
        explicit_group_param_counts = {}
        if explicit_group_assignments:
            grouped_params = {}
            for name, param in param_dict.items():
                group_name = explicit_group_assignments.get(name)
                if group_name is None:
                    continue
                grouped_params.setdefault(group_name, []).append(param)
            for group_name, params in grouped_params.items():
                overrides = module_param_group_overrides.get(group_name, {})
                default_wd = 0.0 if all(
                    name in explicit_nodecay_names
                    for name, assigned_group in explicit_group_assignments.items()
                    if assigned_group == group_name
                ) else weight_decay
                optim_groups.append({
                    'params': params,
                    'weight_decay': overrides.get('weight_decay', default_wd),
                    'lr': overrides.get('lr', learning_rate),
                    'betas': overrides.get('betas', betas),
                    'group_name': group_name,
                })
                explicit_group_param_counts[group_name] = sum(p.numel() for p in params)
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_explicit_nodecay_params = sum(
            param_dict[name].numel() for name in explicit_nodecay_names if name in param_dict
        )
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        if explicit_nodecay_names:
            print(
                f"num explicit no-decay tensors: {len(explicit_nodecay_names)}, "
                f"with {num_explicit_nodecay_params:,} parameters"
            )
        if explicit_group_param_counts:
            for group_name, num_params in explicit_group_param_counts.items():
                group_config = next(
                    group for group in optim_groups if group.get('group_name') == group_name
                )
                print(
                    f"optimizer group {group_name}: "
                    f"{len(group_config['params'])} tensors, {num_params:,} parameters, "
                    f"lr={group_config['lr']}, wd={group_config['weight_decay']}, "
                    f"betas={group_config['betas']}"
                )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
