# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


# ============================ 기본 블록 ============================

class SpatialEncoding(nn.Module):
    def __init__(self, input_size: int, model_size: int, requires_grad: bool = True):
        super().__init__()
        self.se = nn.Parameter(torch.randn(1, 1, input_size, model_size), requires_grad=requires_grad)

    def forward(self):
        return self.se  # [1, 1, I, D]


class Embedding(nn.Module):
    """
    패치 없이 사용: num_patches=1, patch_size = window_size 로 두면 '무패치'와 동일.
    """
    def __init__(self, input_size: int, patch_size: int, model_size: int, dropout: float):
        super().__init__()
        self.encoding = nn.Linear(patch_size, model_size)
        self.spatial_encoding = SpatialEncoding(input_size, model_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B, P, I, S]
        x = self.encoding(x) + self.spatial_encoding()  # [B, P, I, D]
        return self.dropout(x)  # [B, P, I, D]


class Patching(nn.Module):
    def __init__(self, num_patches: int):
        super().__init__()
        self.num_patches = num_patches

    def forward(self, x):  # [B, T, C]
        return einops.rearrange(x, 'b (p s) i -> b p i s', p=self.num_patches)  # [B, P, I, S]


class Unpatching(nn.Module):
    def forward(self, x):  # [B, P, I, S]
        return einops.rearrange(x, 'b p i s -> b (p s) i')  # [B, T, C]


class Attention(nn.Module):
    """
    변수축(I)에서의 멀티헤드 어텐션. s를 외부에서 주면 그것을 사용(프로토타입 실험 시 유용).
    is_diagonal_masked=False 가 기본(전처리 안함).
    """
    def __init__(self, input_size, model_size, n_heads, dropout, bias=True, is_diagonal_masked: bool = False):
        super().__init__()
        assert model_size % n_heads == 0
        self.model_size = model_size
        self.n_heads = n_heads
        self.head_size = model_size // n_heads

        self.Q = nn.Linear(model_size, model_size, bias)
        self.K = nn.Linear(model_size, model_size, bias)
        self.V = nn.Linear(model_size, model_size, bias)
        self.linear = nn.Linear(model_size, model_size)
        self.dropout = nn.Dropout(dropout)

        self.is_diagonal_masked = is_diagonal_masked
        diag_mask = 1.0 - torch.eye(input_size).unsqueeze(0).unsqueeze(0)  # [1, 1, I, I]
        self.register_buffer('diag_mask', diag_mask)

    def forward(self, q, k, v, s: Optional[torch.Tensor] = None):  # q,k,v: [B, I, D]
        B, I, _ = q.size()
        v = self.V(v).view(B, I, self.n_heads, self.head_size)  # [B, I, H, Dh]

        if s is None:
            q = self.Q(q).view(B, I, self.n_heads, self.head_size)  # [B, I, H, Dh]
            k = self.K(k).view(B, I, self.n_heads, self.head_size)  # [B, I, H, Dh]
            scores = torch.einsum('bqhe,bkhe->bhqk', q, k) / math.sqrt(self.head_size)  # [B, H, I, I]
            s = torch.softmax(scores, dim=-1)
            if self.is_diagonal_masked:
                s = s * self.diag_mask  # [B, H, I, I]
                s = s / (s.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            if self.is_diagonal_masked:
                s = s * self.diag_mask
            s = s / (s.sum(dim=-1, keepdim=True) + 1e-6)

        s_d = self.dropout(s)
        attn = torch.einsum('bhql,blhd->bqhd', s_d, v).reshape(B, I, self.model_size)  # [B, I, D]
        return self.linear(attn), s  # [B, I, D], [B, H, I, I]


class SpatialEncoder(nn.Module):
    def __init__(self, input_size, model_size, feedforward_size, num_heads, dropout,
                 bias=True, is_diagonal_masked: bool = False):
        super().__init__()
        self.attn = Attention(input_size, model_size, num_heads, dropout, bias, is_diagonal_masked)
        self.norm1 = nn.LayerNorm(model_size)
        self.norm2 = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(model_size, feedforward_size, bias), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward_size, model_size, bias), nn.Dropout(dropout),
        )

    def forward(self, x, s: Optional[torch.Tensor] = None):  # x: [B, P, I, D] or [B*P, I, D]
        is_4d = (x.dim() == 4)
        if is_4d:
            B, P, I, D = x.shape
            x = x.view(B * P, I, D)  # [B*P, I, D]

        attn, s = self.attn(x, x, x, s)  # attn: [B*P, I, D], s: [B*P, H, I, I]
        x = self.norm1(x + self.dropout(attn))
        x = self.norm2(x + self.ff(x))

        if is_4d:
            x = x.view(B, P, I, D)                    # [B, P, I, D]
            s = s.view(B, P, self.attn.n_heads, I, I) # [B, P, H, I, I]

        return x, s


class Decoder(nn.Module):
    def __init__(self, patch_size, model_size):
        super().__init__()
        self.ln = nn.LayerNorm(model_size)
        self.linear = nn.Linear(model_size, patch_size)

    def forward(self, x):  # [B, P, I, D]
        x = self.ln(x)
        return self.linear(x)  # [B, P, I, S]


# ============================ 프로토타입 뱅크 ============================

class ProtoBank(nn.Module):
    """
    프로토타입 E_k ∈ R^{I×I}, '행 softmax'로 행합=1을 강제.
    - k-means(코사인)로 초기화 가능.
    - 유사도는 cosine(vec(A), vec(E_k)) 사용.
    """
    def __init__(self, num_prototypes: int, input_size: int, temperature: float = 1.0):
        super().__init__()
        self.K = num_prototypes
        self.I = input_size
        self.tau = temperature
        # 로짓 파라미터: R_k ∈ R^{I×I}
        self.R = nn.Parameter(torch.randn(self.K, self.I, self.I))

    def prototypes(self) -> torch.Tensor:
        # 행별 softmax: E_k(i,:) = softmax(R_k(i,:)/tau)
        # if self.tau == 1.0:
        #     E = F.softmax(self.R, dim=-1)
        # else:
        #     E = F.softmax(self.R / self.tau, dim=-1)
        E = self.R
        return E  # [K, I, I], 각 행 합 = 1

    @staticmethod
    def _l2n(x: torch.Tensor, dim: int) -> torch.Tensor:
        return x / (x.norm(p=2, dim=dim, keepdim=True) + 1e-12)
    # ▼▼▼▼▼ [추가] JS Divergence 계산을 위한 static method ▼▼▼▼▼
    @staticmethod
    def _rowwise_js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        p,q: [..., I] (행 확률분포). 반환: [...] (스칼라 JS per-row)
        """
        m = 0.5 * (p + q)
        js = 0.5 * (p.clamp_min(eps) * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1) + \
             0.5 * (q.clamp_min(eps) * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
        return js
    def calculate_similarity(self, attn, metric='cosine', detach_attn=True):
        if detach_attn:
            attn = attn.detach()                       # [B,H,I,I]
        B, H, I, _ = attn.shape
        E_full = self.prototypes().to(attn.device, dtype=attn.dtype)  # [K,I,I]
        K = E_full.size(0)

        if metric == 'cosine':  # ★ flatten cosine
            A = attn.reshape(B, H, -1)                 # [B,H,D]
            E = E_full.reshape(K, -1)                  # [K,D]
            A = self._l2n(A, dim=-1); E = self._l2n(E, dim=-1)
            s = torch.einsum('bhd,kd->bhk', A, E)      # [B,H,K]

        elif metric == 'dot_product':
            A = attn.reshape(B, H, -1); E = E_full.reshape(K, -1)
            s = torch.einsum('bhd,kd->bhk', A, E)      # [B,H,K]

        elif metric == 'mse':
            A = attn.reshape(B, H, -1); E = E_full.reshape(K, -1)
            s = -((A.unsqueeze(2) - E.unsqueeze(0).unsqueeze(0))**2).mean(dim=-1)

        # elif metric in ('cosine_row', 'row_cosine'):   # ★ 행별 cosine (reshape 금지)
        #     # 행 단위 L2 정규화
        #     A_row = attn / (attn.norm(dim=-1, keepdim=True) + 1e-12)      # [B,H,I,I]
        #     E_row = E_full / (E_full.norm(dim=-1, keepdim=True) + 1e-12)  # [K,I,I]
        #     # 브로드캐스트 후 행별 cos: <A[i,:], E[i,:]>
        #     cos_row = (A_row.unsqueeze(2) * E_row.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B,H,K,I]

        #     # 행 집계(평균/Top-k/최대)
        #     k = int(getattr(self, 'row_topk', 0))
        #     if k and k > 0:
        #         k = min(k, I)
        #         if k == 1:
        #             s = cos_row.max(dim=-1).values                          # [B,H,K]
        #         else:
        #             s = torch.topk(cos_row, k=k, dim=-1).values.mean(dim=-1)
        #     else:
        #         s = cos_row.mean(dim=-1)                                    # [B,H,K]

        # elif metric == 'kl':
        #     A_b = attn.unsqueeze(2)                         # [B,H,1,I,I]
        #     E_b = E_full.unsqueeze(0).unsqueeze(0)          # [1,1,K,I,I]
        #     kl_values = F.kl_div((E_b+1e-12).log(), A_b, reduction='none', log_target=False)
        #     s = -kl_values.sum(dim=-1).mean(dim=-1)         # [B,H,K]

        elif metric == 'js':
            A_b = attn.unsqueeze(2)                         # [B,H,1,I,I]
            E_b = E_full.unsqueeze(0).unsqueeze(0)          # [1,1,K,I,I]
            js_per_row = self._rowwise_js_div(A_b, E_b)     # [B,H,K,I]
            s = -js_per_row.mean(dim=-1)                    # [B,H,K]

        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return s
    # ▲▲▲▲▲ [추가] JS Divergence 계산을 위한 static method ▲▲▲▲▲
    # def calculate_similarity(
    #     self,
    #     attn: torch.Tensor,
    #     metric : str = 'cosine',
    #     detach_attn: bool = True
    # ) -> torch.Tensor:
    #     """
    #     attn: [B, H, I, I] (마지막 레이어의 어텐션 맵)
    #     metric: 'cosine', 'dot', 'mse' 중 선택
    #     return: s ∈ R^{B,H,K}
    #     """
    #     if detach_attn:
    #         attn = attn.detach()
            
    #     B, H, I, _ = attn.shape
    #     A = attn.reshape(B, H, -1)
    #     E = self.prototypes().reshape(self.K, -1)
    #     E = E.to(attn.device, dtype=attn.dtype)
        
    #     if metric == 'cosine':
    #         A = self._l2n(A, dim=-1)
    #         E = self._l2n(E, dim=-1)
    #         s = torch.einsum('bhd,kd->bhk', A, E) # [B,H,K]
    #     elif metric in ('cosine_row', 'row_cosine'):
    #         # attn: [B,H,I,I], prototypes: [K,I,I]
    #         B, H, I, _ = attn.shape
    #         E_full = self.prototypes().to(attn.device, dtype=attn.dtype)  # [K,I,I]

    #         # 1) 행 단위 L2 정규화
    #         A_row = attn / (attn.norm(dim=-1, keepdim=True) + 1e-12)              # [B,H,I,I]
    #         E_row = E_full / (E_full.norm(dim=-1, keepdim=True) + 1e-12)          # [K,I,I]

    #         # 2) 브로드캐스트로 행별 cosine: <A[i,:], E[i,:]>
    #         #    결과: [B,H,K,I]  (샘플 B, 헤드 H, 프로토 K, 행 I)
    #         A_b = A_row.unsqueeze(2)                                               # [B,H,1,I,I]
    #         E_b = E_row.unsqueeze(0).unsqueeze(0)                                  # [1,1,K,I,I]
    #         cos_row = (A_b * E_b).sum(dim=-1)                                      # [B,H,K,I]

    #         # 3) 행 집계: 평균 또는 Top-k 평균
    #         k = int(getattr(self, 'row_topk', 0))  # 0/None -> mean, 1 -> max, k>1 -> top-k mean
    #         if k and k > 0:
    #             k = min(k, I)
    #             if k == 1:
    #                 s = cos_row.max(dim=-1).values                                 # [B,H,K]
    #             else:
    #                 topk = torch.topk(cos_row, k=k, dim=-1).values                 # [B,H,K,k]
    #                 s = topk.mean(dim=-1)                                          # [B,H,K]
    #         else:
    #             s = cos_row.mean(dim=-1)                                           # [B,H,K]
    #     elif metric == 'dot_product':
    #         s = torch.einsum('bhd,kd->bhk', A, E) # [B,H,K]
    #     elif metric == 'mse':
    #         # MSE 이므로, 음수 취함
    #         A_exp = A.unsqueeze(2)  # [B,H,1,I*I]
    #         E_exp = E.unsqueeze(0).unsqueeze(0)  # [1,1,K,I*I]
    #         s = -((A_exp - E_exp) ** 2).mean(dim=-1)  # [B,H,K]
    #     elif metric == 'kl':
    #         # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    #         # [수정] 행별(row-wise) KL Divergence를 안정적으로 계산하는 새로운 로직
    #         # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    #         B, H, I, _ = attn.shape
    #         E = self.prototypes().to(attn.device, dtype=attn.dtype)  # [K, I, I]

    #         # 1. 브로드캐스팅을 위해 각 텐서에 K, (B, H)를 위한 빈 차원을 추가합니다.
    #         A_b = attn.unsqueeze(2)      # [B, H, 1, I, I] (K를 위한 공간)
    #         E_b = E.unsqueeze(0).unsqueeze(0)  # [1, 1, K, I, I] (B, H를 위한 공간)

    #         # 2. F.kl_div를 브로드캐스팅된 텐서에 직접 적용합니다.
    #         #    결과는 [B, H, K, I, I] 모양의 텐서가 됩니다.
    #         kl_values = F.kl_div(
    #             (E_b + 1e-12).log(),
    #             A_b,
    #             reduction='none',
    #             log_target=False
    #         )

    #         # 3. 각 행(마지막 차원)에 대한 KL Divergence 값을 합산합니다.
    #         kl_per_row = kl_values.sum(dim=-1)  # [B, H, K, I]

    #         # 4. 한 맵에 있는 모든 행(I개)의 KL Divergence 값의 평균을 냅니다.
    #         kl_per_map = kl_per_row.mean(dim=-1)  # [B, H, K]

    #         # 5. 거리를 유사도로 변환하기 위해 음수를 취합니다.
    #         s = -kl_per_map
            
    #     # ▼▼▼▼▼ [추가] JS Divergence 계산 로직 ▼▼▼▼▼
    #     elif metric == 'js':
    #         B, H, I, _ = attn.shape
    #         E = self.prototypes().to(attn.device, dtype=attn.dtype) # [K,I,I]

    #         # 1. 브로드캐스팅을 위해 텐서 확장
    #         A_b = attn.unsqueeze(2)      # [B, H, 1, I, I]
    #         E_b = E.unsqueeze(0).unsqueeze(0)  # [1, 1, K, I, I]

    #         # 2. 행별 JS Divergence 계산
    #         js_per_row = self._rowwise_js_div(A_b, E_b) # [B, H, K, I]

    #         # 3. 맵(I) 단위로 평균을 내어 최종 거리 계산
    #         js_per_map = js_per_row.mean(dim=-1) # [B, H, K]

    #         # 4. (중요) 거리를 유사도로 변환하기 위해 음수 처리
    #         s = -js_per_map
    #     # ▲▲▲▲▲ [추가] JS Divergence 계산 로직 ▲▲▲▲▲
        
    #     # ▼▼▼▼▼ [수정] JS Divergence 계산 로직 (Top-k 평균 적용) ▼▼▼▼▼
    #     # elif metric == 'js':
    #     #     B, H, I, _ = attn.shape
    #     #     E = self.prototypes().to(attn.device, dtype=attn.dtype) # [K,I,I]

    #     #     # 💡 Top-k 설정. 이 값은 외부에서 파라미터로 받는 것이 좋습니다. (e.g., args.topk_dist)
    #     #     k = 3

    #     #     # 1. 브로드캐스팅을 위해 텐서 확장
    #     #     A_b = attn.unsqueeze(2)      # [B, H, 1, I, I]
    #     #     E_b = E.unsqueeze(0).unsqueeze(0)  # [1, 1, K, I, I]

    #     #     # 2. 행별 JS Divergence 계산 (각 변수의 관계 분포 차이)
    #     #     js_per_row = self._rowwise_js_div(A_b, E_b) # [B, H, K, I]

    #     #     # 3. 각 맵(I) 내에서 불일치도가 가장 큰 Top-k개의 값을 선택
    #     #     if k > 1 and k < I:
    #     #         # torch.topk를 사용하여 가장 큰 k개의 값을 찾습니다.
    #     #         topk_vals, _ = torch.topk(js_per_row, k=k, dim=-1) # [B, H, K, k]
                
    #     #         # 4. Top-k 값들의 평균을 해당 맵의 최종 불일치도(거리)로 사용
    #     #         js_per_map = topk_vals.mean(dim=-1) # [B, H, K]
    #     #     elif k >= I: # k가 변수 개수보다 크거나 같으면 전체 평균과 동일
    #     #         js_per_map = js_per_row.mean(dim=-1)
    #     #     else: # k=1 이면 max와 동일
    #     #         js_per_map = js_per_row.max(dim=-1).values

    #     #     # 5. (중요) 거리를 유사도로 변환하기 위해 음수 처리
    #     #     s = -js_per_map
    #     # ▲▲▲▲▲ [수정] JS Divergence 계산 로직 (Top-k 평균 적용) ▲▲▲▲▲
            
    #     else:
    #         raise ValueError(f"Unsupported metric: {metric}. Choose from 'cosine', 'dot', 'mse', 'kl'")
    #     return s

    @torch.no_grad()
    def init_from_centroids(self, centroids: torch.Tensor):
        """
        centroids: [K, I, I], 각 행 합이 1인 분포(권장). 없으면 내부에서 행 정규화.
        행 softmax 파라미터 역치환: R = tau * log(E)
        """
        assert centroids.shape == (self.K, self.I, self.I)
        # 🔧 디바이스/형 맞춤 + 복사 방식 수정(.data 대입 금지)
        E = centroids.to(self.R.device, dtype=self.R.dtype).clone()
        # E = E / (E.sum(dim=-1, keepdim=True) + 1e-12)
        # E = torch.clamp(E, 1e-6, 1.0)
        # R_new = self.tau * torch.log(E)
        R_new = E
        self.R.copy_(R_new)  # graph/디바이스 보존


# ============================ 본 모델 (무패치 설정) ============================

class SARADProto(nn.Module):
    """
    - 패치 없음: num_patches=1, patch_size=window_size
    - 마지막 레이어 어텐션 s_last ∈ [B,H,I,I] 반환
    - 프로토타입 뱅크(옵션)로 s_logits ∈ [B,H,K], p ∈ [B,H,K], entropy ∈ [B,H] 계산 지원
    """
    def __init__(
        self,
        input_size: int,            # I (= C)
        window_size: int,           # T
        model_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        is_diagonal_masked: bool = False,
        num_prototypes: int = 0,    # K=0 이면 프로토타입 미사용
        proto_temperature: float = 1.0,
        softmax_temperature: float = 10.0,  # 유사도->확률 변환 온도(alpha). 학습 X(고정).
        similarity_metric: str = 'cosine'
    ):
        super().__init__()
        self.I = input_size
        self.T = window_size
        self.D = model_size
        self.L = num_layers
        self.H = num_heads

        # '무패치': P=1, S=T
        self.patching = Patching(num_patches=1)
        patch_size = window_size  # S = T
        self.embedding = Embedding(input_size, patch_size, model_size, dropout)
        self.encoder_layers = nn.ModuleList([
            SpatialEncoder(input_size, model_size, 4 * model_size, num_heads, dropout,
                           True, is_diagonal_masked)
            for _ in range(num_layers)
        ])
        self.decoder = Decoder(patch_size, model_size)
        self.unpatching = Unpatching()

        # 프로토타입
        self.K = num_prototypes
        self.alpha = softmax_temperature
        self.similarity_metric = similarity_metric
        print('self.alpha:',self.alpha)
        self.proto_bank = None
        if self.K > 0:
            self.proto_bank = ProtoBank(num_prototypes, input_size, temperature=proto_temperature)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B,T,C]
        return:
          - x_enc: [B,1,I,D] (디코더 입력)
          - s_all: [B,L,1,H,I,I]
          - s_last: [B,H,I,I]
        """
        x = self.patching(x)            # [B,1,I,T]
        x = self.embedding(x)           # [B,1,I,D]
        s_all = []
        for layer in self.encoder_layers:
            x, s = layer(x)             # s: [B,1,H,I,I]
            s_all.append(s)
        s_all = torch.stack(s_all, dim=1)  # [B,L,1,H,I,I]
        s_last = s_all[:, -1, 0]           # [B,H,I,I]
        return x, s_all, s_last

    def _decode(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(x_enc)    # [B,1,I,T]
        x_hat = self.unpatching(x_hat) # [B,T,C]
        return x_hat

    @staticmethod
    def _entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
        # p: 확률(softmax 출력)
        return -(p * (p.clamp_min(eps)).log()).sum(dim=dim)

    def forward(
        self,
        x: torch.Tensor,
        compute_proto: bool = False,
        detach_attn: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict], torch.Tensor]:
        """
        return:
          x_hat:    [B,T,C]
          s_last:   [B,H,I,I]
          proto:    dict or None (s_logits=[B,H,K], p=[B,H,K], entropy=[B,H], p_batch=[H,K])
          s_all:    [B,L,1,H,I,I]
        """
        x_enc, s_all, s_last = self._encode(x)
        x_hat = self._decode(x_enc)

        proto = None
        if compute_proto and (self.proto_bank is not None):
            # s_logits: [B,H,K]
            s_logits = self.proto_bank.calculate_similarity(s_last, metric=self.similarity_metric,detach_attn=detach_attn)
            # p = F.softmax(self.alpha * s_logits, dim=-1)  # [B,H,K], alpha 고정(학습 X)
            p = F.softmax(s_logits/self.alpha, dim=-1)  # [B,H,K], alpha 고정(학습 X)
            ent = self._entropy(p, dim=-1)                # [B,H]
            p_batch = p.mean(dim=0)                       # [H,K] (equip-loss용 배치 평균)
            proto = dict(s_logits=s_logits, p=p, entropy=ent, p_batch=p_batch)

        return x_hat, s_last, proto, s_all

    @torch.no_grad()
    def init_prototypes_from_centroids(self, centroids: torch.Tensor):
        """
        centroids: [K,I,I]
        - k-means(cosine) 중심을 [K,I,I]로 받아 행 정규화 후 R에 log로 역치환.
        """
        assert self.proto_bank is not None, "num_prototypes=0 인 경우 프로토타입 미사용."
        self.proto_bank.init_from_centroids(centroids)
