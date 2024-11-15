"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.n_gaussians = 42
        C.vocab_size = None
        C.block_size = None
        C.scores_size = None
        C.far_reco_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.0
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config, sample_weights_data=None):
        super().__init__()
        # assert config.vocab_size is not None
        assert config.block_size is not None and config.scores_size is not None and config.far_reco_size is not None
        self.block_size = config.block_size
        self.scores_size = config.scores_size
        self.far_reco_size = config.far_reco_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type]) #very verbose, can simplify


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(1, config.n_embd), # is this stupid? linear layer instead of a token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), # 3 for mu, sigma, weight
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_gaussians * 3, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

        if sample_weights_data is not None:
            self.sample_weights_hist = sample_weights_data[0]
            self.sample_weights_bins = sample_weights_data[1]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, sample_weights_var=None):
        device = idx.device
        b, t = idx.size()

        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx.unsqueeze(-1)) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        output = self.lm_head(x) # (batch_size, n_objects, 3*n_gaussians)
        not_near = self.scores_size + self.far_reco_size
        output = output[:, -not_near:, :] #get rid of all tokens that correspond to near detector
        batch_size, n_objects, n_gaussians = output.shape
        # this is a huge mess
        output = output.reshape(batch_size, n_objects, int(n_gaussians/3), 3)
        scores_output = output[:, :self.scores_size, :, :]
        far_reco_output = output[:, self.scores_size:, :, :]

        #split between scores and far_reco

        scores_mixture = self.compute_mixture(scores_output, transform=True)
        far_reco_mixture = self.compute_mixture(far_reco_output)

        if targets is not None:
            if sample_weights_var is not None:
                # Should do this in init but dont have device there
                sample_weights_hist = torch.tensor(
                    self.sample_weights_hist, dtype=float, device=device
                )
                sample_weights_bins = torch.tensor(
                    self.sample_weights_bins, dtype=float, device=device
                )
                # Finding training sample weights from histogram
                sample_weights = sample_weights_hist[
                    torch.bucketize(sample_weights_var, sample_weights_bins) - 1
                ]
                sample_weights = sample_weights.reshape(sample_weights.shape[0], 1)
                # Apply sigmoid then training sample weights to log_probs to make loss
                scores_loss = -scores_mixture.log_prob(targets[:, :self.scores_size])
                scores_loss = torch.nn.functional.sigmoid(scores_loss) * sample_weights
                scores_loss = scores_loss.mean()
                far_reco_loss = -far_reco_mixture.log_prob(targets[:, self.scores_size:])
                far_reco_loss = torch.nn.functional.sigmoid(far_reco_loss) * sample_weights
                far_reco_loss = far_reco_loss.mean()
                loss = (scores_loss + far_reco_loss) / 2
            else:
                scores_loss = -scores_mixture.log_prob(targets[:, :self.scores_size]).mean()
                far_reco_loss = -far_reco_mixture.log_prob(targets[:, self.scores_size:]).mean()
                loss = scores_loss + far_reco_loss
            return output, loss

        return output

    def compute_mixture(self, output, transform=False):
        mu = output[...,0]
        sigma = torch.exp(output[...,1]) # sigma>0 #paper givt uses softplus
        weights = torch.nn.functional.softmax(output[...,2], dim=-1) # normalize weights
        # temperature scale
        # which mixture component to sample from
        mixture = torch.distributions.Categorical(weights)

        # sample from the mixture component
        components = torch.distributions.Normal(mu, sigma)
        if transform:
            components = torch.distributions.TransformedDistribution(components, torch.distributions.transforms.SigmoidTransform())
        else:
            components = torch.distributions.TransformedDistribution(components, torch.distributions.transforms.ExpTransform())
        # construct the gaussian mixture distribution
        return torch.distributions.MixtureSameFamily(mixture, components)

    @torch.no_grad()
    def total_log_probability(self, idx, targets):
        """
        Computes the total log probability of the targets given the idx
        """
        output = self.forward(idx)
        gaussian_mixture = self.compute_mixture(output)
        return gaussian_mixture.log_prob(targets)

    @torch.no_grad()
    def log_probability(self, idx, targets):
        """
        Computes the conditional log probability of the targets given the idx
        """
        output = self.forward(idx)
        output = output[:,-1, :, :].unsqueeze(1)
        gaussian_mixture = self.compute_mixture(output)
        return gaussian_mixture.log_prob(targets)

    @torch.no_grad()
    def generate(self, idx=None, num_dims=None, temperature=1.0, device='cpu'):
        if num_dims is None:
            num_dims = self.block_size - 1

        start_dim = idx.shape[1]
        x = idx

        inner_idx = 0
        for i in range(start_dim, num_dims):

            output = self.forward(x)
            output = output[:,-1, :, :].unsqueeze(1)
            transform = False
            if inner_idx <= self.scores_size - 1:
                transform = True # first 4 dimensions are scores
            gaussian_mixture = self.compute_mixture(output, transform=transform)

            x_next = gaussian_mixture.sample()
            x = torch.cat((x, x_next), dim=1)

            inner_idx += 1

        return x

