from torch import nn,einsum
from einops import rearrange
def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out
class JointAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.2, proj_drop=0.2,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.weight_audio = nn.Parameter(torch.Tensor([0.8]))
        # self.weight_video = nn.Parameter(torch.Tensor([0.2]))

    def forward(self, high_feature, low_feature):
        h = self.num_heads
        # project feature1 to q1, k1, v1 values
        q1, k1, v1 = self.qkv(high_feature).chunk(3, dim=-1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q1, k1, v1))

        # project feature2 to q2, k2, v2 values
        q2, k2, v2 = self.qkv(low_feature).chunk(3, dim=-1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q2, k2, v2))



        # attention
        out1 = attn(q1, k1, v1)
        out2 = attn(q2, k2, v2)

        # merge back the heads
        out1 = rearrange(out1, '(b h) n d -> b n (h d)', h=h)
        out2 = rearrange(out2, '(b h) n d -> b n (h d)', h=h)
        # 自动调整权重
        # weight_audio = self.weight_audio.sigmoid()
        # weight_video = self.weight_video.sigmoid()
        # combine the outputs of feature1 and feature2
        weight_audio = 0.75
        weight_video = 0.25
        combined_out = out1*weight_audio+out2*weight_video



        low_feature = self.proj(combined_out)
        low_feature = self.proj_drop(low_feature)


        # return the combined output
        return low_feature
class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(input_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs):
        weights = torch.tanh(self.attention_weights(inputs))
        attention_scores = torch.softmax(self.context_vector(weights), dim=1)
        attended_inputs = inputs * attention_scores
        return attended_inputs