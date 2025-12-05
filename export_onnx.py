import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math
import types
import argparse
from learning.models.score_network import ScoreNetMultiPair
from learning.models.refine_network import RefineNet  

# ==============================================================================
# 1. 定义手动实现的 Attention 类 (用于替换 nn.MultiheadAttention)
#    这是为了解决 "Exporting the operator 'aten::_native_multi_head_attention'..." 错误
# ==============================================================================
class ManualMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._qkv_same_embed_dim = False 
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        tgt_len, bsz, embed_dim = query.shape
        
        if torch.equal(query, key) and torch.equal(key, value):
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            qkv = qkv.view(tgt_len, bsz, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3)
            if self.in_proj_bias is not None:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            else:
                b_q = b_k = b_v = None
            q = F.linear(query, w_q, b_q).view(tgt_len, bsz, self.num_heads, self.head_dim)
            k = F.linear(key, w_k, b_k).view(tgt_len, bsz, self.num_heads, self.head_dim)
            v = F.linear(value, w_v, b_v).view(tgt_len, bsz, self.num_heads, self.head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores += attn_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, None

# ==============================================================================
# 2. 定义修复版的 extract_feat 和 forward 方法
#    这是为了解决 "aten::unflatten" 和 reshape 错误
# ==============================================================================
def patched_extract_feat(self, A, B):
    """ 替换原模型中的 extract_feat，修复 reshape/permute 问题 """
    bs = A.shape[0]  # B*L
    x = torch.cat([A,B], dim=0)
    x = self.encoderA(x)
    a = x[:bs]
    b = x[bs:]
    ab = torch.cat((a,b), dim=1)
    ab = self.encoderAB(ab)
    
    # --- 修复逻辑开始 ---
    # 原代码: ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))
    # 修复为 ONNX 友好的操作:
    B_L, C_prime, H_prime, W_prime = ab.shape
    ab_flat = ab.view(B_L, C_prime, H_prime * W_prime)
    ab_permuted = ab_flat.transpose(1, 2) # (B, HW, C)
    ab = self.pos_embed(ab_permuted)
    # --- 修复逻辑结束 ---

    ab, _ = self.att(ab, ab, ab)
    
    # 修复最后的 reshape
    # 原代码: return ab.mean(dim=1).reshape(bs,-1)
    return ab.mean(dim=1).view(bs, -1)

def patched_forward_score(self, A, B, L):
    output = {}
    # 注意：在 ONNX 导出时，bs 是符号，不能直接用于 reshape
    # 但我们通过 view 和明确的维度计算来规避
    bs = A.shape[0] // L 
    feats = self.extract_feat(A, B)   #(B*L, C)
    
    # --- 修复逻辑开始 ---
    # 原代码: x = feats.reshape(bs,L,-1)
    feat_dim = feats.shape[-1]
    x = feats.view(bs, L, feat_dim)
    x, _ = self.att_cross(x, x, x)
    output['score_logit'] = self.linear(x).view(bs, L)
    return output

# ==============================================================================
# 3. RefineNet 修复补丁 (新增)
# ==============================================================================
def patched_forward_refine(self, A, B):
    """ 替换 RefineNet 中的 forward，修复 reshape/permute 问题 """
    # 原代码: bs = len(A) -> 在 ONNX 中 len() 可能有问题，改用 shape[0]
    bs = A.shape[0]
    output = {}

    x = torch.cat([A,B], dim=0)
    x = self.encodeA(x)
    a = x[:bs]
    b = x[bs:]

    ab = torch.cat((a,b),1).contiguous()
    ab = self.encodeAB(ab)  #(B,C,H,W)

    # --- 修复逻辑开始 ---
    # 原代码: ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))
    B_dim, C_prime, H_prime, W_prime = ab.shape
    ab_flat = ab.view(B_dim, C_prime, H_prime * W_prime)
    ab_permuted = ab_flat.transpose(1, 2) # (B, HW, C)
    ab = self.pos_embed(ab_permuted)
    # --- 修复逻辑结束 ---

    output['trans'] = self.trans_head(ab).mean(dim=1)
    output['rot'] = self.rot_head(ab).mean(dim=1)

    return output

# ==============================================================================
# 3. 辅助函数：替换模型中的 Attention 层
# ==============================================================================
def replace_attention_layers(module):
    # 递归替换所有子模块中的 MultiheadAttention
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            print(f"  -> 正在替换层: {name} (MultiheadAttention) 为 ManualMultiheadAttention")
            new_att = ManualMultiheadAttention(
                embed_dim=child.embed_dim, 
                num_heads=child.num_heads, 
                bias=(child.in_proj_bias is not None), 
                batch_first=child.batch_first
            )
            # 复制权重 (参数名称完全一致，可以直接加载)
            new_att.load_state_dict(child.state_dict())
            # 替换
            setattr(module, name, new_att)
        else:
            replace_attention_layers(child)

def load_config(weight_dir):
    config_path = os.path.join(weight_dir, 'config.yml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# --- 新增辅助类：解决 RefineNet 混用 cfg.key 和 cfg['key'] 的问题 ---
class HybridConfig(dict):
    """ 
    一个简单的包装类，让字典支持属性访问 (obj.key)。
    同时保留字典的下标访问 (obj['key']) 能力。
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'HybridConfig' object has no attribute '{name}'")
# -------------------------------------------------------------------

# ==============================================================================
# 5. 导出函数
# ==============================================================================
SCORE_WEIGHT_DIR = 'weights/2024-01-11-20-02-45'
REFINE_WEIGHT_DIR = 'weights/2023-10-28-18-33-37' # <--- 请确认此路径是否正确

def export_score_model():
    print(f"\n=== 开始导出 Score 模型 (内存热修补模式) ===")
    weight_dir = SCORE_WEIGHT_DIR
    onnx_path = os.path.join(weight_dir, 'score_model.onnx')
    
    # 1. 加载配置
    cfg_dict = load_config(weight_dir)
    cfg = argparse.Namespace(**cfg_dict)
    c_in = getattr(cfg, 'c_in', 6)
    
    # 2. 初始化原始模型
    print("初始化原始模型...")
    model = ScoreNetMultiPair(cfg=cfg, c_in=c_in)
    
    # 3. 加载权重
    ckpt_path = os.path.join(weight_dir, 'model_best.pth')
    print(f"加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    
    # 4. 【关键步骤】在内存中修补模型
    print("正在应用内存补丁以修复 ONNX 兼容性...")
    
    # 4.1 替换 Attention 层
    replace_attention_layers(model)
    
    # 4.2 替换方法 (Method Swapping)
    # 使用 types.MethodType 将我们定义的修复版函数绑定到模型实例上
    model.extract_feat = types.MethodType(patched_extract_feat, model)
    model.forward = types.MethodType(patched_forward_score, model)
    
    model.eval()
    model.cuda()

    # 5. 创建虚拟输入
    bs = 1
    L = 3
    h, w = getattr(cfg, 'input_resize', [160, 160])
    dummy_input_A = torch.randn(bs * L, c_in, h, w).cuda()
    dummy_input_B = torch.randn(bs * L, c_in, h, w).cuda()
    
    print(f"输入形状: A={dummy_input_A.shape}, B={dummy_input_B.shape}, L={L}")

    # 6. 导出 ONNX
    print(f"正在导出到: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_input_A, dummy_input_B, L),
        onnx_path,
        input_names=['input_A', 'input_B', 'L'],
        output_names=['score_logit'],
        dynamic_axes={
            'input_A': {0: 'batch_times_L'},
            'input_B': {0: 'batch_times_L'},
            'score_logit': {0: 'batch_size'}
        },
        opset_version=13,
        do_constant_folding=True
    )
    print(f"✅ Score 模型导出成功！路径: {onnx_path}")

def export_refine_model():
    print(f"\n=== 开始导出 Refine 模型 (内存热修补模式) ===")
    weight_dir = REFINE_WEIGHT_DIR
    onnx_path = os.path.join(weight_dir, 'refine_model.onnx')
    
    if not os.path.exists(weight_dir):
        print(f"❌ 错误: Refine 权重目录不存在: {weight_dir}")
        return

    cfg_dict = load_config(weight_dir)
    # cfg = argparse.Namespace(**cfg_dict)
    # c_in = getattr(cfg, 'c_in', 6)
    cfg = HybridConfig(cfg_dict)
    c_in = cfg_dict.get('c_in', 6)
    
    print("初始化原始模型...")
    model = RefineNet(cfg=cfg, c_in=c_in)
    
    ckpt_path = os.path.join(weight_dir, 'model_best.pth')
    print(f"加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    
    print("正在应用内存补丁...")
    # 1. 替换 Attention 层 (RefineNet 使用了 TransformerEncoderLayer，内部有 MultiheadAttention)
    replace_attention_layers(model)
    
    # 2. 替换 forward 方法
    model.forward = types.MethodType(patched_forward_refine, model)
    
    model.eval()
    model.cuda()

    bs = 1
    h, w = cfg_dict.get('input_resize', [160, 160])
    dummy_input_A = torch.randn(bs, c_in, h, w).cuda()
    dummy_input_B = torch.randn(bs, c_in, h, w).cuda()
    
    print(f"输入形状: A={dummy_input_A.shape}, B={dummy_input_B.shape}")
    print(f"正在导出到: {onnx_path}")
    
    torch.onnx.export(
        model,
        (dummy_input_A, dummy_input_B),
        onnx_path,
        input_names=['input_A', 'input_B'],
        output_names=['trans', 'rot'],
        dynamic_axes={
            'input_A': {0: 'batch_size'},
            'input_B': {0: 'batch_size'},
            'trans': {0: 'batch_size'},
            'rot': {0: 'batch_size'}
        },
        opset_version=13,
        do_constant_folding=True
    )
    print(f"✅ Refine 模型导出成功！路径: {onnx_path}")

if __name__ == "__main__":
    if not os.path.exists(SCORE_WEIGHT_DIR):
        print(f"错误: 权重目录不存在: {SCORE_WEIGHT_DIR}")
        sys.exit(1)

    try:
        export_score_model()
        export_refine_model()
        print("\n🎉 所有模型导出完成！")
    except Exception as e:
        print(f"\n❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()