# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.metric_utils import calculate_all_metrics, calculate_range_metrics
import pickle
import matplotlib.pyplot as plt    # === 컬러바: 우측에 외부 축을 만들어 정렬 + 간결한 tick ===
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import h5py

# === eval.py 하단(또는 evaluate() 위쪽) 유틸 추가 ===
@torch.no_grad()
def visualize_attn_with_prototypes(
    s_last: torch.Tensor, s_logits: torch.Tensor, proto_bank,
    out_dir: str, b_index: int = 0, topk: int = 3,
    similarity_metric: str = "cosine", fname_prefix: str = "vis",
    sample_label: str = None,
    title_prefix: str = None,
):
    """
    한 배치에서 샘플(b_index) 하나를 선택해,
    1행: 헤드별 원본 어텐션 맵
    2..(1+topk)행: 각 헤드의 최근접 프로토타입 Top-k
    컬러바는 그리드 우측 외부에 1개만 배치.
    """
    import os
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    device = s_last.device
    E = proto_bank.prototypes().to(device=device, dtype=s_last.dtype)  # [K,I,I]
    B, H, I, _ = s_last.shape
    assert 0 <= b_index < B
    os.makedirs(out_dir, exist_ok=True)

    # --- 거리 계산: d_bhK (작을수록 가까움)
    s_bhK = s_logits[b_index]  # [H,K]
    if similarity_metric == "js":
        d_bhK = -s_bhK
    else:
        d_bhK = (1.0 - s_bhK).clamp_min(0.0)

    topk = int(min(topk, d_bhK.shape[-1]))
    vals, idxs = torch.topk(d_bhK, k=topk, dim=-1, largest=False)  # [H, topk]

    # --- 레이아웃: 행=(1+topk), 열=H
    nrows, ncols = 1 + topk, H
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(2.2*ncols, 2.2*nrows), squeeze=False)

    # 1행: 원본 어텐션
    last_im = None
    for h in range(H):
        ax = axes[0, h]
        last_im = ax.imshow(
            s_last[b_index, h].detach().cpu().numpy(),
            vmin=0.0, vmax=1.0, aspect='auto'
        )
        ax.set_title(f"Head {h}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # 2..(1+topk)행: 헤드별 Top-k 프로토타입
    for h in range(H):
        for r in range(topk):
            ax = axes[1+r, h]
            k = int(idxs[h, r].item())
            d = float(vals[h, r].item())
            last_im = ax.imshow(
                E[k].detach().cpu().numpy(),
                vmin=0.0, vmax=1.0, aspect='auto'
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"P{k}  d={d:.4f}", fontsize=7)

    # ----- 제목 & 그리드 정리 먼저 -----
    ttl_left = f"b={b_index}, top{topk}"
    ttl_right = f"label={sample_label}" if sample_label is not None else ""
    ttl_head = f"{title_prefix}  " if title_prefix else ""
    fig.suptitle(f"{ttl_head}{ttl_left}   {ttl_right}", fontsize=10)

    # 오른쪽에 컬러바가 들어갈 여백을 확보(rect의 right를 0.98보다 더 줄여줌)
    plt.tight_layout(rect=[0.0, 0.0, 0.96, 0.96])

    # ----- 이제 '그리드 바깥 오른쪽'에 컬러바 전용 축(cax) 생성 -----
    # 마지막 열 맨 위 axes의 위치를 기준으로 figure 좌표에서 계산
    ref_ax = axes[0, -1]
    pos = ref_ax.get_position()  # Bbox in figure coordinates
    gap = 0.006                  # 그리드와 컬러바 사이 간격
    cbar_width = 0.012           # 컬러바 폭
    cax = fig.add_axes([pos.x1 + gap, pos.y0, cbar_width, pos.height])

    cbar = fig.colorbar(last_im, cax=cax)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.locator = MaxNLocator(nbins=4, prune='both')   # tick 개수 축약
    cbar.formatter = FormatStrFormatter('%.2f')         # 0.00 형식
    cbar.update_ticks()

    out_path = os.path.join(out_dir, f"{fname_prefix}_b{b_index}_top{topk}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

@torch.no_grad()
def compute_elementwise_score(
    attn_last: torch.Tensor,         # [B,H,I,I]  (마지막 레이어 어텐션)
    s_logits: torch.Tensor,          # [B,H,K]    (헤드별 프로토타입 유사도 로그릿)
    proto_bank,                      # model.proto_bank (prototypes() -> [K,I,I])
    recon_diag: torch.Tensor,        # [B,I]      (센서별 MSE; time 평균)
) -> torch.Tensor:
    """
    1) (b,h)별 최상 유사도 프로토타입 선택
    2) 행(센서) 단위 cosine 유사도: A[i,:] vs E[i,:] → [B,H,I]
    3) 헤드 평균: [B,I]
    4) softmax(센서축)로 가중치 W: [B,I]
    5) (W ⊙ recon_diag).sum(-1): [B] 최종 스코어
    """
    assert attn_last.dim() == 4, f"expect [B,H,I,I], got {attn_last.shape}"
    B, H, I, _ = attn_last.shape
    assert recon_diag.shape == (B, I), f"recon_diag shape must be [B,I], got {recon_diag.shape}"

    # (1) (b,h)별 최상 프로토타입 index
    k_idx = s_logits.argmax(dim=-1)               # [B,H]

    # (2) 선택된 프로토타입 행렬
    E = proto_bank.prototypes().to(attn_last.device, dtype=attn_last.dtype)  # [K,I,I]
    E_sel = E[k_idx.reshape(-1)].view(B, H, I, I)                            # [B,H,I,I]

    # (3) 행(센서) 단위 cosine 유사도
    A_row = attn_last
    A_row = A_row / (A_row.norm(p=2, dim=-1, keepdim=True) + 1e-12)         # [B,H,I,I]
    E_row = E_sel   / (E_sel.norm(p=2, dim=-1, keepdim=True) + 1e-12)       # [B,H,I,I]
    v = (A_row * E_row).sum(dim=-1)                                         # [B,H,I] (센서별 cos sim)

    # (4) 헤드 평균 → 샘플별 센서 유사도
    V = v.mean(dim=1)                                                       # [B,I]

    Vn = 1-V  # 클수록 '비유사' (0~2)
    # (5) 센서축 softmax로 가중치
    W = F.softmax(Vn, dim=-1)                                                # [B,I]

    # (6) 가중합 (element-wise 곱 후 센서합)
    scores = (W * recon_diag).sum(dim=-1)                                   # [B]
    return scores

# ---------------------------------------------------------------------
# 프로토타입 기반 채널 진단: "가장 가까운 프로토타입"과의 행별 JS divergence
# attn: [B,H,I,I], proto["p"]: [B,H,K], E: [K,I,I] (행-softmax 보장)
# return detec_diag: [B,I]
# ---------------------------------------------------------------------
def _rowwise_js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    p,q: [..., I] (행 확률분포). 반환: [...] (스칼라 JS per-row)
    """
    m = 0.5 * (p + q)
    js = 0.5 * (p.clamp_min(eps) * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1) + \
         0.5 * (q.clamp_min(eps) * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
    return js  # [...]

@torch.no_grad()
def compute_detec_diag(attn: torch.Tensor, proto: dict, proto_bank, eps: float = 1e-12):
    """
    attn:      [B,H,I,I]  (last-layer attention)
    proto:     dict with 'p' ∈ [B,H,K] (soft assignment)
    proto_bank: model.proto_bank (provides prototypes() -> [K,I,I])
    return:
      detec_diag: [B,I]  (변수별 진단 점수; 클수록 '프로토타입과 불일치' = 이상)
    """
    assert attn.dim() == 4, f"expect [B,H,I,I], got {attn.shape}"
    B, H, I, _ = attn.shape
    P = proto["p"]                    # [B,H,K]
    k_idx = P.argmax(dim=-1)          # [B,H]

    E = proto_bank.prototypes()       # [K,I,I]
    E = E.to(attn.device)

    # (b,h)별로 프로토타입 선택 → [B,H,I,I]
    E_sel = E[k_idx.reshape(-1)].view(B, H, I, I)

    # 행별(JS per-row): 각 변수 i에 대해 분포 비교
    js_rows = _rowwise_js_div(attn, E_sel)   # [B,H,I]
    detec_diag = js_rows.mean(dim=1)         # head-mean → [B,I]
    return detec_diag


# ---------------------------------------------------------------------
# 내부 정규화 버퍼 보장
# ---------------------------------------------------------------------
def _ensure_norm_buffers(model: nn.Module, num_channels: int):
    """
    recon_avg/std, detec_avg/std: scalar buffers
    rdiag_avg/std, ddiag_avg/std: [C] buffers
    """
    def _reg(name, shape):
        if not hasattr(model, name):
            model.register_buffer(name, torch.zeros(shape, dtype=torch.float32))
        else:
            buf = getattr(model, name)
            if tuple(buf.shape) != tuple(shape):
                # 재등록(형상 불일치 시)
                delattr(model, name)
                model.register_buffer(name, torch.zeros(shape, dtype=torch.float32))

    _reg("recon_avg", ())
    _reg("recon_std", ())
    _reg("detec_avg", ())
    _reg("detec_std", ())
    _reg("rdiag_avg", (num_channels,))
    _reg("rdiag_std", (num_channels,))
    _reg("ddiag_avg", (num_channels,))
    _reg("ddiag_std", (num_channels,))
    
    _reg("sim_avg", ())
    _reg("sim_std", ())


# ---------------------------------------------------------------------
# Evaluate (엔트로피 스칼라 + JS 진단)
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, test_loader, args):
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    all_labels, all_scores, all_diag = [], [], []

    alpha = getattr(args, "alpha", 1.0)
    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    similarity_metric = getattr(args, "similarity_metric", "cosine")

    C_inferred = None

    for batch_i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.numpy()

        # 모델 추론: 프로토 경로 ON, 어텐션 detach
        x_hat, s_last, proto, _ = model(inputs, compute_proto=True, detach_attn=True)  # x_hat:[B,T,C], s_last:[B,H,I,I]
        if C_inferred is None:
            C_inferred = x_hat.shape[-1]
            _ensure_norm_buffers(model, C_inferred)

        # --- Reconstruction ---
        rec = criterion(x_hat, inputs)           # [B, T, C]
        recon_diag = rec.mean(dim=1)             # [B, C]
        recon_loss = recon_diag.mean(dim=1)      # [B]

        # --- Detection ---
        # 스칼라: 엔트로피(헤드 평균)
        # det_entropy = proto["entropy"].mean(dim=1)   # [B]
        # 채널 진단: JS(row-wise) vs 가장 가까운 프로토타입
        detec_diag = compute_detec_diag(s_last, proto, model.proto_bank)  # [B, C]
        # (참고) 스칼라 진단이 필요하다면 per-channel 평균:
        # detec_loss = detec_diag.mean(dim=1)          # [B]
        # === evaluate() 반복문 안, detec_diag까지 계산된 직후에 삽입 ===
        if getattr(args, "vis_attn_with_proto", False) and (batch_i % 10 == 0):
            vis_dir = os.path.join(args.log_dir, "vis_attn_proto", "test")
            topk_for_vis = int(getattr(args, "topk_for_vis", 3))
            b_index = 0  # 미니배치 첫 샘플만 시각화
            y_str = str(labels[b_index])   # ← 단 한 줄로 끝
            try:
                out_img = visualize_attn_with_prototypes(
                    s_last=s_last,
                    s_logits=proto["s_logits"],
                    proto_bank=model.proto_bank,
                    out_dir=vis_dir,
                    b_index=b_index,
                    topk=topk_for_vis,
                    similarity_metric=getattr(args, "similarity_metric", "cosine"),
                    fname_prefix=f"batch{len(all_scores)}_y={y_str}",
                    sample_label=y_str,         # ← 그대로 제목에 표시
                )
                # print(f"[vis] saved → {out_img}")
            except Exception as e:
                print(f"[vis] skip due to error: {e}")
        # --- Normalize (model buffers 사용) ---
        recon_score = ((recon_loss.cpu() - model.recon_avg.cpu()) / (model.recon_std.cpu() + 1e-12)).numpy()
        # detec_score = ((det_entropy.cpu() - model.detec_avg.cpu()) / (model.detec_std.cpu() + 1e-12)).numpy()
        
        # 두 함수 공통으로
        score_mode = getattr(args, "score_mode", "recon")  # "recon" | "recon+sim"
        if score_mode == "elementwise":
            # ✨ 새 모드: 헤드별 최다유사 프로토타입 기반 행유사도 × 센서별 MSE
            elem_scores = compute_elementwise_score(
                attn_last=s_last,                    # [B,H,I,I]
                s_logits=proto["s_logits"],          # [B,H,K]
                proto_bank=model.proto_bank,         # prototypes(): [K,I,I]
                recon_diag=recon_diag,               # [B,I]
            )
            scores = elem_scores.cpu().numpy()       # [B]
        elif score_mode == "recon":
            scores = recon_score
            
        elif score_mode == "recon+sim" and args.reweight==False:
            max_sim = proto['s_logits'].max(dim=-1).values.mean(dim=1) # [B]
            if getattr(args, "similarity_metric", "cosine") == 'js':
                d = -max_sim  # s가 -js_per_map이므로, d는 js_per_map이 됨
            else: # cosine 등 다른 유사도
                d = (1.0 - max_sim).clamp_min(0.0)
            sim_score = ((d.cpu() - model.sim_avg.cpu()) / (model.sim_std.cpu() + 1e-12)).numpy()
            
            scores = recon_score + sim_score
            
            
            if args.only_sim_score:
                scores = sim_score
        elif score_mode == "recon+sim" and args.reweight==True:
            s = proto['s_logits']                              # [B,H,K], cosine similarity
            B, H, K = s.shape
            M = min(getattr(args, "nb_of_proto", 5), K)        # top-M proto
            tau = float(getattr(args, "reweight_tau", 1.0))

            # # 1) cosine → distance
            # d = (1.0 - s).clamp_min(0.0)                       # [B,H,K]
            if getattr(args, "similarity_metric", "cosine") == 'js':
                d = -s  # s가 -js_per_map이므로, d는 js_per_map이 됨
            else: # cosine 등 다른 유사도
                d = (1.0 - s).clamp_min(0.0)
            # 2) head별 top-M 거리 선택 (가까운 순)
            vals, idxs = torch.topk(d, k=M, dim=-1, largest=False)# vals: [B,H,M]

            # 3) 최근접 거리
            d1 = vals[..., 0]                                  # [B,H]

            # 4) reweighting factor (논문 식 근사: softmax(-d/tau))
            probs_pos = torch.softmax(vals / tau, dim=-1)         # [B,H,M]
            p_closest = probs_pos[..., 0]                       # [B,H] # 최근접(가장 작은 d)의 확률
            weights = 1.0 - p_closest                           # [B,H]

            # 5) 헤드별 reweighted score (정규화 전)
            head_score = d1 * weights                          # [B,H]

            # 6) 헤드 집계 (PatchCore는 patch-wise max, 그래서 head-wise max)
            raw_score = head_score.max(dim=1).values           # [B]

            # 7) 이제 validation에서 sim_avg, sim_std도 **reweighting 포함한 raw_score**
            #    분포로 미리 구해둬야 함.
            sim_avg = model.sim_avg.to(raw_score.device)       # scalar or [1]
            sim_std = model.sim_std.to(raw_score.device)
            sim_score = ((raw_score - sim_avg) / (sim_std + 1e-12)).cpu().numpy()
            
            scores = recon_score + sim_score
            if args.only_sim_score:
                scores = sim_score

        
        # scores = recon_score + detec_score

        recon_diag_norm = ((recon_diag.cpu() - model.rdiag_avg.cpu()) / (model.rdiag_std.cpu() + 1e-12)).numpy()
        detec_diag_norm = ((detec_diag.cpu() - model.ddiag_avg.cpu()) / (model.ddiag_std.cpu() + 1e-12)).numpy()
        # diagno = recon_diag_norm + alpha * detec_diag_norm  # [B, C]
        diagno = recon_diag_norm + detec_diag_norm  # [B, C]

        all_scores.extend(scores)
        all_labels.extend(labels)
        all_diag.extend(diagno)
        

    all_scores = np.asarray(all_scores, dtype=np.float32)
    all_labels = np.asarray(all_labels, dtype=np.int32)
    all_diag   = np.asarray(all_diag,   dtype=np.float32)

    metrics = calculate_all_metrics(all_labels, all_scores)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    return metrics, all_diag


# ---------------------------------------------------------------------
# 정규화 통계 추정 (val set): recon + entropy(스칼라), rdiag + JS 진단(채널)
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_norm_stats(model, val_loader, args):
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")

    recon_losses, det_entropy_list = [], []
    rdiags, ddiags = [], []
    sim_scores = []

    C_inferred = None

    for x, _ in val_loader:
        x = x.to(device)
        x_hat, s_last, proto, _ = model(x, compute_proto=True, detach_attn=True)

        if C_inferred is None:
            C_inferred = x.shape[-1]
            _ensure_norm_buffers(model, C_inferred)

        # --- Reconstruction ---
        rec = criterion(x_hat, x)                 # [B, T, C]
        recon_loss = rec.mean(dim=1).mean(dim=1)  # [B]
        recon_diag = rec.mean(dim=1)              # [B, C]

        # --- Detection ---
        # det_entropy = proto["entropy"].mean(dim=1)                      # [B]
        detec_diag  = compute_detec_diag(s_last, proto, model.proto_bank)  # [B, C]
        
        if args.use_sim_score and args.reweight==False:
            # --- 가장 가까운 프로토타입과 similairty : 1-max_k cos_sim ---
            max_sim = proto['s_logits'].max(dim=-1).values.mean(dim=1) # [B]
            if getattr(args, "similarity_metric", "cosine") == 'js':
                d = -max_sim  # s가 -js_per_map이므로, d는 js_per_map이 됨
            else: # cosine 등 다른 유사도
                d = (1.0 - max_sim).clamp_min(0.0)
            sim_scores.append(d.cpu())
        if args.use_sim_score and args.reweight==True:
            s = proto['s_logits']                              # [B,H,K]
            B, H, K = s.shape
            M = min(getattr(args, "nb_of_proto", 5), K)        # top-M proto
            tau = float(getattr(args, "reweight_tau", 1.0))

            # # 1) cosine → distance
            # d = (1.0 - s).clamp_min(0.0)                       # [B,H,K]
            if getattr(args, "similarity_metric", "cosine") == 'js':
                d = -s  # s가 -js_per_map이므로, d는 js_per_map이 됨
            else: # cosine 등 다른 유사도
                d = (1.0 - s).clamp_min(0.0)
            # 2) head별 top-M 거리 선택 (가까운 순)
            vals, idxs = torch.topk(d, k=M, dim=-1, largest=False)   # vals: [B,H,M]

            # 3) 최근접 거리
            d1 = vals[..., 0]                                  # [B,H]

            # 4) reweighting factor (논문 식 근사: softmax(-d/tau))
            probs_pos = torch.softmax(vals / tau, dim=-1)         # [B,H,M]
            p_closest = probs_pos[..., 0]                       # [B,H] # 최근접(가장 작은 d)의 확률
            weights = 1.0 - p_closest                           # [B,H]

            # 5) 헤드별 reweighted score (정규화 전)
            head_score = d1 * weights                          # [B,H]

            # 6) 헤드 집계 (PatchCore는 patch-wise max, 여기선 head-wise max 권장)
            raw_score = head_score.max(dim=1).values
            sim_scores.append(raw_score.cpu())

        recon_losses.append(recon_loss.cpu())
        # det_entropy_list.append(det_entropy.cpu())
        rdiags.append(recon_diag.cpu())
        ddiags.append(detec_diag.cpu())

    # concatenate
    recon_losses = torch.cat(recon_losses)              # [N]
    # det_entropy_list = torch.cat(det_entropy_list)      # [N]
    rdiags = torch.cat(rdiags)                          # [N, C]
    ddiags = torch.cat(ddiags)                          # [N, C]
    if args.use_sim_score:
        sim_scores = torch.cat(sim_scores)                  # [N]

    # register to model buffers
    model.recon_avg.copy_(recon_losses.mean())
    model.recon_std.copy_(recon_losses.std() + 1e-12)
    # model.detec_avg.copy_(det_entropy_list.mean())
    # model.detec_std.copy_(det_entropy_list.std() + 1e-12)
    model.rdiag_avg.copy_(rdiags.mean(dim=0))
    model.rdiag_std.copy_(rdiags.std(dim=0) + 1e-12)
    model.ddiag_avg.copy_(ddiags.mean(dim=0))
    model.ddiag_std.copy_(ddiags.std(dim=0) + 1e-12)
    
    if args.use_sim_score:
        model.sim_avg.copy_(sim_scores.mean())
        model.sim_std.copy_(sim_scores.std() + 1e-12)

    print('[norm_stats registered to model buffers]')
    # ▼▼▼▼▼ 추가할 디버깅 코드 ▼▼▼▼▼
    print("="*50)
    print(f"Normalization Stats Check for {args.model}")
    if hasattr(model, 'recon_avg'):
        print(f"Reconstruction Score Avg: {model.recon_avg.item():.4f}")
        print(f"Reconstruction Score Std: {model.recon_std.item():.4f}")
    if hasattr(model, 'sim_avg'):
        print(f"Similarity Score Avg: {model.sim_avg.item():.4f}")
        print(f"Similarity Score Std: {model.sim_std.item():.4f}")
    print("="*50)



import os
import numpy as np
import torch
from torch import nn
from utils.metric_utils import calculate_all_metrics
import pickle

# ---------------------------------------------------------------------
# Proto-entropy + row-wise JS 진단 버전 저장형 평가
#  - model.compute_proto=True / detach_attn=True 경로 사용
#  - 정규화는 (1) norm_stats dict 제공 시 그 값 사용
#             (2) 미제공 시 model의 buffer(recon_avg 등) 사용
#  - 저장 항목: scores, labels, recon_scores, detec_scores, attn_idx, attn(s_last)
#  - 반환: (metrics, all_diag)  (diag는 저장 X)
# ---------------------------------------------------------------------
from typing import Optional, Tuple, Dict, Any
NormStats = Dict[str, Tuple[np.ndarray, np.ndarray]]

@torch.no_grad()
def evaluate_and_save_proto(
    model,
    test_loader,
    args,
    save_dir: str,
    save_name: str = "scores",
    save_format: str = "npz",      # "npz" | "pkl"
    save_attention: bool = True,   # s_last 저장 여부
    attn_stride: int = 500,        # 글로벌 스텝 간격
    norm_stats: Optional[NormStats] = None # {'recon':(avg,std),'detec':(avg,std),'rdiag':(avg,std),'ddiag':(avg,std)}
):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    similarity_metric = getattr(args, "similarity_metric", "cosine")

    def _to_numpy_local(t):
        return t.cpu().numpy() if isinstance(t, torch.Tensor) else t

    # --- 정규화 통계 로딩 (dict 우선, 없으면 model buffer 사용) ---
    if norm_stats is None:
        # model buffer 기반
        recon_avg = _to_numpy_local(getattr(model, "recon_avg"))
        recon_std = _to_numpy_local(getattr(model, "recon_std"))
        detec_avg = _to_numpy_local(getattr(model, "detec_avg"))
        detec_std = _to_numpy_local(getattr(model, "detec_std"))
        rdiag_avg = _to_numpy_local(getattr(model, "rdiag_avg"))
        rdiag_std = _to_numpy_local(getattr(model, "rdiag_std"))
        ddiag_avg = _to_numpy_local(getattr(model, "ddiag_avg"))
        ddiag_std = _to_numpy_local(getattr(model, "ddiag_std"))
    else:
        recon_avg, recon_std = map(_to_numpy_local, norm_stats['recon'])
        detec_avg, detec_std = map(_to_numpy_local, norm_stats['detec'])
        rdiag_avg, rdiag_std = map(_to_numpy_local, norm_stats['rdiag'])
        ddiag_avg, ddiag_std = map(_to_numpy_local, norm_stats['ddiag'])

    alpha  = getattr(args, "alpha", 1.0)
    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # --- 누적 버퍼 ---
    all_labels = []
    all_scores = []
    all_diag   = []   # 저장 X (반환만)
    all_recon_scores = []
    all_detec_scores = []
    all_sim_scores = []

    attn_idx, attn_list = [], []
    cursor = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = _to_numpy_local(labels)

        # 모델 추론: x_hat:[B,T,C], s_last:[B,H,I,I], proto: {'p':[B,H,K], 'entropy':[B,H], ...}
        x_hat, s_last, proto, _ = model(inputs, compute_proto=True, detach_attn=True)

        # --- Reconstruction ---
        rec = criterion(x_hat, inputs)         # [B, T, C]
        recon_diag = rec.mean(dim=1)           # [B, C]
        recon_loss = recon_diag.mean(dim=1)    # [B]

        # --- Detection (스칼라: proto entropy / 채널: row-wise JS against nearest prototype) ---
        # det_entropy = proto["entropy"].mean(dim=1)  # [B]
        detec_diag  = compute_detec_diag(s_last, proto, model.proto_bank)  # [B, C]

        # --- Normalize (scalars per sample) ---
        recon_score = (recon_loss.cpu().numpy() - recon_avg) / (recon_std + 1e-12)
        # detec_score = (det_entropy.cpu().numpy() - detec_avg) / (detec_std + 1e-12)
        
        # 두 함수 공통으로
        score_mode = getattr(args, "score_mode", "recon")  # "recon" | "recon+sim" | "elementwise"
        if score_mode == "elementwise":
            # ✨ 새 모드: 헤드별 최다유사 프로토타입 기반 행유사도 × 센서별 MSE
            elem_scores = compute_elementwise_score(
                attn_last=s_last,                    # [B,H,I,I]
                s_logits=proto["s_logits"],          # [B,H,K]
                proto_bank=model.proto_bank,         # prototypes(): [K,I,I]
                recon_diag=recon_diag,               # [B,I]
            )
            scores = elem_scores.cpu().numpy()       # [B]
        elif score_mode == "recon":
            scores = recon_score
            
        elif score_mode == "recon+sim" and args.reweight==False:
            max_sim = proto['s_logits'].max(dim=-1).values.mean(dim=1) # [B]
            if getattr(args, "similarity_metric", "cosine") == 'js':
                d = -max_sim  # s가 -js_per_map이므로, d는 js_per_map이 됨
            else: # cosine 등 다른 유사도
                d = (1.0 - max_sim).clamp_min(0.0)
            sim_score = ((d.cpu() - model.sim_avg.cpu()) / (model.sim_std.cpu() + 1e-12)).numpy()
            
            scores = recon_score + sim_score
            
            if args.only_sim_score:
                scores = sim_score
        elif score_mode == "recon+sim" and args.reweight==True:
            s = proto['s_logits']                              # [B,H,K], cosine similarity
            B, H, K = s.shape
            M = min(getattr(args, "nb_of_proto", 5), K)        # top-M proto
            tau = float(getattr(args, "reweight_tau", 1.0))

            # # 1) cosine → distance
            # d = (1.0 - s).clamp_min(0.0)                       # [B,H,K]
            if getattr(args, "similarity_metric", "cosine") == 'js':
                d = -s  # s가 -js_per_map이므로, d는 js_per_map이 됨
            else: # cosine 등 다른 유사도
                d = (1.0 - s).clamp_min(0.0)
            # 2) head별 top-M 거리 선택 (가까운 순)
            vals, idxs = torch.topk(d, k=M, dim=-1, largest=False)# vals: [B,H,M]

            # 3) 최근접 거리
            d1 = vals[..., 0]                                  # [B,H]

            # 4) reweighting factor (논문 식 근사: softmax(-d/tau))
            probs_pos = torch.softmax(vals / tau, dim=-1)         # [B,H,M]
            p_closest = probs_pos[..., 0]                       # [B,H] # 최근접(가장 작은 d)의 확률
            weights = 1.0 - p_closest                           # [B,H]

            # 5) 헤드별 reweighted score (정규화 전)
            head_score = d1 * weights                          # [B,H]

            # 6) 헤드 집계 (PatchCore는 patch-wise max, 그래서 head-wise max)
            raw_score = head_score.max(dim=1).values           # [B]

            # 7) 이제 validation에서 sim_avg, sim_std도 **reweighting 포함한 raw_score**
            #    분포로 미리 구해둬야 함.
            sim_avg = model.sim_avg.to(raw_score.device)       # scalar or [1]
            sim_std = model.sim_std.to(raw_score.device)
            sim_score = ((raw_score - sim_avg) / (sim_std + 1e-12)).cpu().numpy()
            
            scores = recon_score + sim_score
            
            if args.only_sim_score:
                scores = sim_score

        # --- Normalize (per-channel diag; 저장 X) ---
        recon_diag_norm = (recon_diag.cpu().numpy() - rdiag_avg) / (rdiag_std + 1e-12)  # [B, C]
        detec_diag_norm = (detec_diag.cpu().numpy() - ddiag_avg) / (ddiag_std + 1e-12)  # [B, C]
        diagno = recon_diag_norm + alpha * detec_diag_norm

        # 누적
        all_scores.extend(scores)
        all_labels.extend(labels)
        all_diag.extend(diagno)
        all_recon_scores.extend(recon_score)
        # all_detec_scores.extend(detec_score)
        if args.use_sim_score:
            all_sim_scores.extend(sim_score)

        # 어텐션(s_last) 샘플링 저장
        if save_attention and (cursor % attn_stride == 0) and (s_last is not None):
            attn_idx.append(cursor)
            attn_list.append(s_last.detach().cpu().numpy())  # [B,H,I,I]

        cursor += inputs.size(0)

    # numpy 변환
    all_scores = np.asarray(all_scores, dtype=np.float32)
    all_labels = np.asarray(all_labels, dtype=np.int32)
    # all_diag   = np.asarray(all_diag,   dtype=np.float32)
    all_recon_scores = np.asarray(all_recon_scores, dtype=np.float32)
    # all_detec_scores = np.asarray(all_detec_scores, dtype=np.float32)
    all_sim_scores = np.asarray(all_sim_scores, dtype=np.float32)

    # 지표
    metrics = calculate_all_metrics(all_labels, all_scores)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 저장
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, save_name)

    if save_format.lower() == "npz":
        np.savez_compressed(
            base + ".npz",
            labels=all_labels,
            recon_scores=all_recon_scores,
            sim_scores=all_sim_scores,
            attn_idx=np.asarray(attn_idx, dtype=np.int32),
            attn=np.asarray(attn_list, dtype=object),  # 가변 batch 대응
        )
    elif save_format.lower() == "pkl":
        payload = dict(
            labels=all_labels,
            recon_scores=all_recon_scores,
            sim_scores=all_sim_scores,
            attn_idx=np.asarray(attn_idx, dtype=np.int32),
            attn=attn_list,
        )
        with open(base + ".pkl", "wb") as f:
            pickle.dump(payload, f)
    elif save_format.lower() == "h5":
        with h5py.File(base + ".h5", "w") as f:
            f.create_dataset("labels", data=all_labels, compression="gzip")
            f.create_dataset("predictions", data=x_hat.cpu().numpy(), compression="gzip")
            f.create_dataset("recon_scores", data=all_recon_scores, compression="gzip")
            f.create_dataset("sim_scores", data=all_sim_scores, compression="gzip")
            f.create_dataset("attn_idx", data=np.asarray(attn_idx, dtype=np.int32))
            # 어텐션은 크기가 커서 개별로 저장하는 것이 안전
            # for i, attn in enumerate(attn_list):
            #     f.create_dataset(f"attn/{i}", data=attn, compression="gzip")
    else:
        raise ValueError(f"Unsupported save_format: {save_format}")

    return metrics, all_diag
import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from typing import List, Tuple, Optional
# @torch.no_grad()
# def auto_focus_dump_segmented(
#     model,
#     test_loader,
#     args,
#     out_dir: str,                    # 디렉터리. 세그먼트별로 개별 h5 저장
#     detect_by: str = "labels",       # "labels" | "scores_auto"
#     score_mode: str = None,
#     q_high: float = 0.995,
#     margin_left: int = 200,
#     margin_right: int = 200,
#     stride: int = 50,
#     topk: int = 3,
#     train_size_for_abs: Optional[int] = None,
# ):
#     """
#     SARADProto.forward(x, compute_proto=..., detach_attn=...) 사용 버전.
#     1) 전체 스코어/라벨 수집 → 이상 구간 산출
#     2) 구간별로 stride 샘플만 다시 돌며 세그먼트 단위 H5에 스트리밍 저장
#     """
#     import os, h5py, numpy as np, torch
#     import torch.nn as nn
#     from typing import List, Tuple

#     assert train_size_for_abs is not None, "train_size_for_abs(len(train)) 필요"
#     os.makedirs(out_dir, exist_ok=True)

#     device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
#     similarity_metric = getattr(args, "similarity_metric", "cosine")
#     score_mode = score_mode or getattr(args, "score_mode", "recon")

#     model.eval()
#     criterion = nn.MSELoss(reduction="none")

#     # 정규화 통계
#     recon_avg = getattr(model, "recon_avg", torch.tensor(0.0)).detach().cpu()
#     recon_std = getattr(model, "recon_std", torch.tensor(1.0)).detach().cpu()
#     sim_avg   = getattr(model, "sim_avg",   torch.tensor(0.0)).detach().cpu()
#     sim_std   = getattr(model, "sim_std",   torch.tensor(1.0)).detach().cpu()

#     def _compute_score(x_hat, inputs, proto, s_last):
#         # recon
#         rec = criterion(x_hat, inputs).mean(dim=1).mean(dim=1)  # [B]
#         recon_score = ((rec.detach().cpu() - recon_avg) / (recon_std + 1e-12)).numpy()

#         if score_mode == "recon":
#             return recon_score

#         # sim
#         if (proto is None) or ("s_logits" not in proto):
#             return recon_score
#         s_logits = proto["s_logits"]  # [B,H,K]
#         if similarity_metric == "js":
#             d = -s_logits
#         else:
#             d = (1.0 - s_logits).clamp_min(0.0)
#         d1 = d.min(dim=-1).values.mean(dim=1).detach().cpu().numpy()  # [B]
#         sim_score = ((d1 - sim_avg.numpy()) / (sim_std.numpy() + 1e-12))
#         if score_mode in ("recon+sim", "elementwise"):
#             return recon_score + sim_score
#         return recon_score

#     # ---------------- 1PASS: score/label 수집 ----------------
#     all_scores_list: List[np.ndarray] = []
#     all_labels_list: List[np.ndarray] = []

#     need_proto = (score_mode != "recon") and (getattr(model, "proto_bank", None) is not None) and (model.K > 0)

#     for inputs, labels in test_loader:
#         inputs = inputs.to(device, non_blocking=True)
#         # SARADProto 시그니처: (x, compute_proto=False, detach_attn=True)
#         x_hat, s_last, proto, _ = model(inputs, compute_proto=need_proto, detach_attn=True)
#         scores = _compute_score(x_hat, inputs, proto, s_last)
#         all_scores_list.append(scores.astype(np.float32))
#         all_labels_list.append(labels.numpy().astype(np.int32))

#         del x_hat, s_last, proto, inputs
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     all_scores = np.concatenate(all_scores_list, axis=0)  # [T_test]
#     all_labels = np.concatenate(all_labels_list, axis=0)  # [T_test]
#     del all_scores_list, all_labels_list

#     # 연속 구간 추출
#     def _ranges_from_mask(mask: np.ndarray) -> List[Tuple[int,int]]:
#         ranges = []
#         N = mask.shape[0]; i = 0
#         while i < N:
#             if mask[i] == 1:
#                 j = i
#                 while j+1 < N and mask[j+1] == 1:
#                     j += 1
#                 ranges.append((i, j))
#                 i = j + 1
#             else:
#                 i += 1
#         return ranges

#     if detect_by == "labels" and (all_labels.sum() > 0):
#         base_ranges = _ranges_from_mask((all_labels > 0).astype(np.int32))
#     else:
#         thr = float(np.quantile(all_scores, q_high))
#         base_ranges = _ranges_from_mask((all_scores >= thr).astype(np.int32))

#     if len(base_ranges) == 0:
#         print("[auto_focus_dump_segmented] no anomaly ranges found.")
#         return

#     # ---------------- 세그먼트별 저장 ----------------
#     for seg_id, (a, b) in enumerate(base_ranges):
#         left = max(0, a - int(margin_left))
#         right = b + int(margin_right)
#         step = max(1, int(stride))
#         focus_rel_list = list(range(left, right + 1, step))
#         if not focus_rel_list:
#             continue

#         out_h5_path = os.path.join(out_dir, f"segment_{seg_id}_a{a}_b{b}.h5")
#         with h5py.File(out_h5_path, "w") as f:
#             g = f.create_group("auto_focus")
#             d_global = g.create_dataset("global_idx", shape=(0,), maxshape=(None,), dtype="int64")
#             d_rel    = g.create_dataset("test_rel_idx", shape=(0,), maxshape=(None,), dtype="int64")
#             d_attn = d_topk_idx = d_topk_dist = None

#             cursor_test = 0
#             cursor_out  = 0

#             for inputs, _labels in test_loader:
#                 inputs = inputs.to(device, non_blocking=True)
#                 x_hat, s_last, proto, _ = model(inputs, compute_proto=need_proto, detach_attn=True)
#                 B = inputs.size(0)

#                 rel_idx_full = np.arange(cursor_test, cursor_test + B, dtype=np.int64)
#                 mask_keep = np.isin(rel_idx_full, focus_rel_list)
#                 if mask_keep.any():
#                     keep_idx_local = torch.as_tensor(np.nonzero(mask_keep)[0], device=inputs.device, dtype=torch.long)
#                     s_pick = s_last.index_select(0, keep_idx_local)  # [M,H,I,I]

#                     if need_proto and (proto is not None) and ("s_logits" in proto):
#                         s_logits_full = proto["s_logits"].index_select(0, keep_idx_local)  # [M,H,K]
#                         if similarity_metric == "js":
#                             d = -s_logits_full
#                         else:
#                             d = (1.0 - s_logits_full).clamp_min(0.0)
#                         k_use = min(int(topk), d.size(-1))
#                         vals, idxs = torch.topk(d, k=k_use, dim=-1, largest=False)  # [M,H,k_use]
#                     else:
#                         M, H, I, _ = s_pick.shape
#                         k_use = int(topk)
#                         vals = torch.full((M, H, k_use), float("nan"), device=inputs.device)
#                         idxs = torch.full((M, H, k_use), -1, device=inputs.device)

#                     rel_idx_kept = rel_idx_full[mask_keep]                  # [M]
#                     abs_idx_kept = rel_idx_kept + int(train_size_for_abs)   # [M]
#                     M = rel_idx_kept.shape[0]

#                     s_np    = s_pick.detach().cpu().numpy().astype(np.float32)
#                     idxs_np = idxs.detach().cpu().numpy().astype(np.int32)
#                     vals_np = vals.detach().cpu().numpy().astype(np.float32)
#                     rel_np  = rel_idx_kept.astype(np.int64)
#                     abs_np  = abs_idx_kept.astype(np.int64)

#                     if d_attn is None:
#                         d_attn = g.create_dataset("attn",      shape=s_np.shape,    maxshape=(None,)+s_np.shape[1:],    dtype="float32")
#                         d_topk_idx  = g.create_dataset("topk_idx",  shape=idxs_np.shape, maxshape=(None,)+idxs_np.shape[1:], dtype="int32")
#                         d_topk_dist = g.create_dataset("topk_dist", shape=vals_np.shape, maxshape=(None,)+vals_np.shape[1:], dtype="float32")
#                         d_global.resize((M,)); d_rel.resize((M,))
#                         d_global[:] = abs_np; d_rel[:] = rel_np
#                         d_attn[:] = s_np; d_topk_idx[:] = idxs_np; d_topk_dist[:] = vals_np
#                         cursor_out = M
#                     else:
#                         d_global.resize((cursor_out+M,)); d_rel.resize((cursor_out+M,))
#                         d_attn.resize((cursor_out+M,)+d_attn.shape[1:])
#                         d_topk_idx.resize((cursor_out+M,)+d_topk_idx.shape[1:])
#                         d_topk_dist.resize((cursor_out+M,)+d_topk_dist.shape[1:])
#                         d_global[cursor_out:cursor_out+M]    = abs_np
#                         d_rel[cursor_out:cursor_out+M]       = rel_np
#                         d_attn[cursor_out:cursor_out+M]      = s_np
#                         d_topk_idx[cursor_out:cursor_out+M]  = idxs_np
#                         d_topk_dist[cursor_out:cursor_out+M] = vals_np
#                         cursor_out += M

#                 cursor_test += B
#                 del x_hat, s_last, proto, inputs
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()

#         print(f"[auto_focus_dump_segmented] segment {seg_id} ({a}-{b}) saved -> {out_h5_path}")

#     print("[auto_focus_dump_segmented] all segments done.")


# @torch.no_grad()
# def auto_focus_dump_segmented(
#     model,
#     test_loader,
#     args,
#     out_dir: str,                    # 디렉터리. 세그먼트별로 개별 h5 저장
#     detect_by: str = "labels",       # "labels" | "scores_auto"
#     score_mode: str = None,
#     q_high: float = 0.995,
#     margin_left: int = 200,
#     margin_right: int = 200,
#     stride: int = 50,
#     topk: int = 3,
#     train_size_for_abs: Optional[int] = None,
# ):
#     """
#     [개선/메모리절약/요구사항충족 버전]
#     - 세그먼트 H5에 다음을 '반드시' 저장:
#         /auto_focus/attn           (N,H,I,I)                ← 실제 어텐션
#         /auto_focus/nn_proto       (N,H,I,I)                ← 각 샘플·헤드의 최근접 프로토 행렬
#         /auto_focus/nn_idx         (N,H) int32              ← 최근접 프로토 인덱스
#         /auto_focus/nn_dist        (N,H) float32            ← 최근접 프로토 거리
#         /auto_focus/sim_d1         (N,) float32             ← 헤드 평균 최근접 거리  (raw)
#         /auto_focus/sim_d1_norm    (N,) float32             ← (sim_d1 - sim_avg)/sim_std
#         /auto_focus/recon          (N,) float32             ← 재구성오차(스칼라, raw)
#         /auto_focus/recon_norm     (N,) float32             ← (recon - recon_avg)/recon_std
#         /auto_focus/score_total    (N,) float32             ← score_mode에 따른 최종 스코어
#         /auto_focus/topk_idx       (N,H,k) int32            ← 상위 k개 프로토 인덱스
#         /auto_focus/topk_dist      (N,H,k) float32          ← 상위 k개 프로토 거리
#         /auto_focus/global_idx     (N,) int64
#         /auto_focus/test_rel_idx   (N,) int64
#         /auto_focus/labels         (N,) int8                ← 가능한 경우
#         /auto_focus/prototypes     (K,I,I) float32          ← 프로토타입 뱅크(파일당 1회)

#     - 1PASS: 점수/라벨로 이상구간 파악
#     - 2PASS: 해당 구간의 stride 지점만 골라, 배치 단위로 H5에 'append' (메모리 누적 방지)
#     """
#     assert train_size_for_abs is not None, "train_size_for_abs(len(train)) 필요"
#     os.makedirs(out_dir, exist_ok=True)

#     device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
#     similarity_metric = getattr(args, "similarity_metric", "cosine")
#     score_mode = score_mode or getattr(args, "score_mode", "recon")

#     model.eval()
#     criterion = nn.MSELoss(reduction="none")

#     # 정규화 통계 (이미 compute_norm_stats로 등록됐다고 가정)
#     recon_avg = getattr(model, "recon_avg", torch.tensor(0.0)).detach().cpu()
#     recon_std = getattr(model, "recon_std", torch.tensor(1.0)).detach().cpu()
#     sim_avg   = getattr(model, "sim_avg",   torch.tensor(0.0)).detach().cpu()
#     sim_std   = getattr(model, "sim_std",   torch.tensor(1.0)).detach().cpu()

#     # --- 스코어 계산 유틸 (1PASS용 스칼라)
#     def _compute_score(x_hat, inputs, proto, s_last):
#         # recon
#         rec = criterion(x_hat, inputs).mean(dim=1).mean(dim=1)  # [B]
#         recon_score = ((rec.detach().cpu() - recon_avg) / (recon_std + 1e-12)).numpy()

#         if score_mode == "recon":
#             return recon_score

#         # sim
#         if (proto is None) or ("s_logits" not in proto):
#             return recon_score
#         s_logits = proto["s_logits"]  # [B,H,K]
#         if similarity_metric == "js":
#             d = -s_logits
#         else:
#             d = (1.0 - s_logits).clamp_min(0.0)
#         d1 = d.min(dim=-1).values.mean(dim=1).detach().cpu().numpy()  # [B] (헤드 평균 최근접 거리)
#         sim_score = ((d1 - sim_avg.numpy()) / (sim_std.numpy() + 1e-12))

#         if score_mode in ("recon+sim", "elementwise"):
#             return (recon_score + sim_score)
#         return recon_score

#     # ---------------- 1PASS: score/label 수집 ----------------
#     all_scores_list: List[np.ndarray] = []
#     all_labels_list: List[np.ndarray] = []

#     # 1PASS에서는 score 계산만 필요하므로, proto 사용 여부는 score_mode에 따라
#     need_proto_first = (score_mode != "recon") and (getattr(model, "proto_bank", None) is not None) and (model.K > 0)

#     for inputs, labels in test_loader:
#         inputs = inputs.to(device, non_blocking=True)
#         x_hat, s_last, proto, _ = model(inputs, compute_proto=need_proto_first, detach_attn=True)
#         scores = _compute_score(x_hat, inputs, proto, s_last)
#         all_scores_list.append(scores.astype(np.float32))
#         all_labels_list.append(labels.numpy().astype(np.int32))

#         del x_hat, s_last, proto, inputs
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     all_scores = np.concatenate(all_scores_list, axis=0)  # [T_test]
#     all_labels = np.concatenate(all_labels_list, axis=0)  # [T_test]
#     del all_scores_list, all_labels_list

#     # ---------------- 이상 구간 추출 ----------------
#     def _ranges_from_mask(mask: np.ndarray) -> List[Tuple[int,int]]:
#         ranges = []
#         N = mask.shape[0]; i = 0
#         while i < N:
#             if mask[i] == 1:
#                 j = i
#                 while j+1 < N and mask[j+1] == 1:
#                     j += 1
#                 ranges.append((i, j))
#                 i = j + 1
#             else:
#                 i += 1
#         return ranges

#     if detect_by == "labels" and (all_labels.sum() > 0):
#         base_ranges = _ranges_from_mask((all_labels > 0).astype(np.int32))
#     else:
#         thr = float(np.quantile(all_scores, q_high))
#         base_ranges = _ranges_from_mask((all_scores >= thr).astype(np.int32))

#     if len(base_ranges) == 0:
#         print("[auto_focus_dump_segmented] no anomaly ranges found.")
#         return

#     # ---------------- 2PASS: 세그먼트별 저장 ----------------
#     # 두 번째 패스에서는 반드시 프로토타입 관련 산출/저장을 수행
#     need_proto_second = (getattr(model, "proto_bank", None) is not None) and (model.K > 0)

#     # 세그먼트별 반복
#     for seg_id, (a, b) in enumerate(base_ranges):
#         left = max(0, a - int(margin_left))
#         right = b + int(margin_right)
#         step = max(1, int(stride))
#         focus_rel_list = list(range(left, right + 1, step))
#         if not focus_rel_list:
#             continue

#         out_h5_path = os.path.join(out_dir, f"segment_{seg_id}_a{a}_b{b}.h5")
#         with h5py.File(out_h5_path, "w") as f:
#             g = f.create_group("auto_focus")

#             # ---- (파일당 1회) 프로토타입 뱅크 저장
#             if need_proto_second:
#                 E = model.proto_bank.prototypes().detach().cpu().numpy().astype("float32")  # (K,I,I)
#                 g.create_dataset("prototypes", data=E, compression="gzip")
#                 g.attrs["proto_K"] = int(E.shape[0])
#                 g.attrs["proto_I"] = int(E.shape[1])
#                 # 캐시(flatten+norm) for fallback sim
#                 E_flat = E.reshape(E.shape[0], -1)
#                 E_flat = E_flat / (np.linalg.norm(E_flat, axis=1, keepdims=True) + 1e-12)
#             else:
#                 E = None
#                 E_flat = None

#             # ---- 메타/세팅 기록
#             g.attrs["similarity_metric"] = str(similarity_metric)
#             g.attrs["score_mode"] = str(score_mode)
#             g.attrs["margin_left"] = int(margin_left)
#             g.attrs["margin_right"] = int(margin_right)
#             g.attrs["stride"] = int(stride)
#             g.attrs["q_high"] = float(q_high)
#             g.attrs["topk"] = int(topk)
#             g.attrs["recon_avg"] = float(recon_avg.item()) if recon_avg.ndim == 0 else float(recon_avg.numpy())
#             g.attrs["recon_std"] = float(recon_std.item()) if recon_std.ndim == 0 else float(recon_std.numpy())
#             g.attrs["sim_avg"] = float(sim_avg.item()) if sim_avg.ndim == 0 else float(sim_avg.numpy())
#             g.attrs["sim_std"] = float(sim_std.item()) if sim_std.ndim == 0 else float(sim_std.numpy())

#             # ---- 인덱스/기본 DS 만들기 (확장 가능)
#             d_global = g.create_dataset("global_idx",   shape=(0,), maxshape=(None,), dtype="int64")
#             d_rel    = g.create_dataset("test_rel_idx", shape=(0,), maxshape=(None,), dtype="int64")
#             d_labels = g.create_dataset("labels",       shape=(0,), maxshape=(None,), dtype="int8")

#             d_attn = d_topk_idx = d_topk_dist = None
#             d_nn_idx = d_nn_dist = d_nn_proto = None
#             d_sim_d1 = d_sim_d1_norm = None
#             d_recon = d_recon_norm = None
#             d_score_total = None

#             cursor_test = 0
#             cursor_out  = 0

#             # 세그먼트마다 test 전체 재순회
#             for inputs, _labels in test_loader:
#                 inputs = inputs.to(device, non_blocking=True)

#                 # 반드시 proto 계산: 최근접 프로토/거리 저장 위해
#                 x_hat, s_last, proto, _ = model(inputs, compute_proto=need_proto_second, detach_attn=True)
#                 B = inputs.size(0)

#                 rel_idx_full = np.arange(cursor_test, cursor_test + B, dtype=np.int64)
#                 mask_keep = np.isin(rel_idx_full, focus_rel_list)

#                 if mask_keep.any():
#                     keep_idx_local = torch.as_tensor(np.nonzero(mask_keep)[0],
#                                                      device=inputs.device, dtype=torch.long)

#                     # 선택 행들
#                     s_pick = s_last.index_select(0, keep_idx_local)  # [M,H,I,I]
#                     M, H, I, _ = s_pick.shape

#                     # --- 재구성 오차 스칼라 (선택행만)
#                     rec_full = criterion(x_hat, inputs).mean(dim=1).mean(dim=1)  # [B]
#                     rec_sel  = rec_full.index_select(0, keep_idx_local)          # [M]
#                     recon_np = rec_sel.detach().cpu().numpy().astype(np.float32) # raw
#                     recon_norm_np = ((recon_np - float(recon_avg.numpy())) /
#                                      (float(sim_max(1.0*recon_std.numpy()))))  # 안전 처리
#                     # 위 한 줄에서 sim_max는 아래 헬퍼 사용 (division by ~0 방지)
#                     # 아래서 정의

#                     # --- 프로토타입 거리/인덱스/최근접 프로토 행렬
#                     if need_proto_second and (proto is not None) and ("s_logits" in proto):
#                         s_logits_full = proto["s_logits"].index_select(0, keep_idx_local)  # [M,H,K]
#                         if similarity_metric == "js":
#                             d = -s_logits_full
#                         else:
#                             d = (1.0 - s_logits_full).clamp_min(0.0)                        # [M,H,K]

#                         k_use = min(int(topk), d.size(-1))
#                         vals, idxs = torch.topk(d, k=k_use, dim=-1, largest=False)         # [M,H,k]
#                         nn_idx_t = idxs[..., 0]                                            # [M,H]
#                         nn_dist_t = vals[..., 0]                                           # [M,H]

#                         # 샘플별 헤드 평균 거리 (raw)
#                         sim_d1_np = nn_dist_t.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [M]
#                         # 정규화
#                         sim_d1_norm_np = ((sim_d1_np - float(sim_avg.numpy())) /
#                                           (float(sim_max(1.0*sim_std.numpy())))).astype(np.float32)

#                         # 최근접 프로토타입 행렬 복원 (CPU NumPy)
#                         nn_idx_np  = nn_idx_t.detach().cpu().numpy().astype(np.int32)      # [M,H]
#                         nn_dist_np = nn_dist_t.detach().cpu().numpy().astype(np.float32)   # [M,H]
#                         if E is not None:
#                             # gather: [M,H,I,I]
#                             nn_proto_np = np.zeros((M, H, I, I), dtype=np.float32)
#                             for m in range(M):
#                                 nn_proto_np[m] = E[nn_idx_np[m]]  # (H,I,I), 헤드마다 다른 인덱스
#                         else:
#                             nn_proto_np = np.full((M, H, I, I), np.nan, dtype=np.float32)

#                         # (옵션) top-k 저장 (요구사항 포함)
#                         topk_idx_np  = idxs.detach().cpu().numpy().astype(np.int32)       # [M,H,k]
#                         topk_dist_np = vals.detach().cpu().numpy().astype(np.float32)     # [M,H,k]
#                     else:
#                         # 프로토 없음 → NaN/-1 채움
#                         k_use = int(topk)
#                         nn_idx_np   = -np.ones((M, H), dtype=np.int32)
#                         nn_dist_np  = np.full((M, H), np.nan, dtype=np.float32)
#                         sim_d1_np       = np.full((M,), np.nan, dtype=np.float32)
#                         sim_d1_norm_np  = np.full((M,), np.nan, dtype=np.float32)
#                         nn_proto_np = np.full((M, H, I, I), np.nan, dtype=np.float32)
#                         topk_idx_np  = -np.ones((M, H, k_use), dtype=np.int32)
#                         topk_dist_np = np.full((M, H, k_use), np.nan, dtype=np.float32)

#                     # --- 최종 스코어(표시용)
#                     if score_mode == "recon":
#                         score_total_np = recon_norm_np.astype(np.float32)
#                     elif score_mode in ("recon+sim", "elementwise"):
#                         score_total_np = (recon_norm_np + sim_d1_norm_np).astype(np.float32)
#                     else:
#                         # 안전 폴백
#                         score_total_np = recon_norm_np.astype(np.float32)

#                     # --- 인덱스/라벨
#                     rel_idx_kept = rel_idx_full[mask_keep]                            # [M]
#                     abs_idx_kept = rel_idx_kept + int(train_size_for_abs)             # [M]
#                     labels_np    = _labels.numpy().astype(np.int8)[mask_keep]         # [M]

#                     rel_np  = rel_idx_kept.astype(np.int64)
#                     abs_np  = abs_idx_kept.astype(np.int64)

#                     # --- 어텐션/탑k CPU로
#                     s_np     = s_pick.detach().cpu().numpy().astype(np.float32)       # [M,H,I,I]

#                     # --- H5 append (처음엔 dataset 생성)
#                     if d_attn is None:
#                         # 가변 길이 DS 생성
#                         d_attn      = g.create_dataset("attn",      shape=s_np.shape,            maxshape=(None,)+s_np.shape[1:],            dtype="float32", compression="gzip")
#                         d_nn_proto  = g.create_dataset("nn_proto",  shape=nn_proto_np.shape,     maxshape=(None,)+nn_proto_np.shape[1:],     dtype="float32", compression="gzip")
#                         d_nn_idx    = g.create_dataset("nn_idx",    shape=nn_idx_np.shape,       maxshape=(None,)+nn_idx_np.shape[1:],       dtype="int32",   compression="gzip")
#                         d_nn_dist   = g.create_dataset("nn_dist",   shape=nn_dist_np.shape,      maxshape=(None,)+nn_dist_np.shape[1:],      dtype="float32", compression="gzip")
#                         d_topk_idx  = g.create_dataset("topk_idx",  shape=topk_idx_np.shape,     maxshape=(None,)+topk_idx_np.shape[1:],     dtype="int32",   compression="gzip")
#                         d_topk_dist = g.create_dataset("topk_dist", shape=topk_dist_np.shape,    maxshape=(None,)+topk_dist_np.shape[1:],    dtype="float32", compression="gzip")

#                         d_sim_d1       = g.create_dataset("sim_d1",       shape=sim_d1_np.shape,      maxshape=(None,), dtype="float32", compression="gzip")
#                         d_sim_d1_norm  = g.create_dataset("sim_d1_norm",  shape=sim_d1_norm_np.shape, maxshape=(None,), dtype="float32", compression="gzip")
#                         d_recon        = g.create_dataset("recon",        shape=recon_np.shape,       maxshape=(None,), dtype="float32", compression="gzip")
#                         d_recon_norm   = g.create_dataset("recon_norm",   shape=recon_norm_np.shape,  maxshape=(None,), dtype="float32", compression="gzip")
#                         d_score_total  = g.create_dataset("score_total",  shape=score_total_np.shape, maxshape=(None,), dtype="float32", compression="gzip")

#                         d_global.resize((M,)); d_rel.resize((M,)); d_labels.resize((M,))
#                         d_global[:] = abs_np; d_rel[:] = rel_np; d_labels[:] = labels_np

#                         d_attn[:]      = s_np
#                         d_nn_proto[:]  = nn_proto_np
#                         d_nn_idx[:]    = nn_idx_np
#                         d_nn_dist[:]   = nn_dist_np
#                         d_topk_idx[:]  = topk_idx_np
#                         d_topk_dist[:] = topk_dist_np

#                         d_sim_d1[:]       = sim_d1_np
#                         d_sim_d1_norm[:]  = sim_d1_norm_np
#                         d_recon[:]        = recon_np
#                         d_recon_norm[:]   = recon_norm_np
#                         d_score_total[:]  = score_total_np

#                         cursor_out = M
#                     else:
#                         # 뒤에 붙이기 (resize 후 슬라이스 대입)
#                         d_global.resize((cursor_out+M,)); d_rel.resize((cursor_out+M,)); d_labels.resize((cursor_out+M,))
#                         d_attn.resize((cursor_out+M,)+d_attn.shape[1:])
#                         d_nn_proto.resize((cursor_out+M,)+d_nn_proto.shape[1:])
#                         d_nn_idx.resize((cursor_out+M,)+d_nn_idx.shape[1:])
#                         d_nn_dist.resize((cursor_out+M,)+d_nn_dist.shape[1:])
#                         d_topk_idx.resize((cursor_out+M,)+d_topk_idx.shape[1:])
#                         d_topk_dist.resize((cursor_out+M,)+d_topk_dist.shape[1:])
#                         d_sim_d1.resize((cursor_out+M,))
#                         d_sim_d1_norm.resize((cursor_out+M,))
#                         d_recon.resize((cursor_out+M,))
#                         d_recon_norm.resize((cursor_out+M,))
#                         d_score_total.resize((cursor_out+M,))

#                         d_global[cursor_out:cursor_out+M] = abs_np
#                         d_rel[cursor_out:cursor_out+M]    = rel_np
#                         d_labels[cursor_out:cursor_out+M] = labels_np

#                         d_attn[cursor_out:cursor_out+M]      = s_np
#                         d_nn_proto[cursor_out:cursor_out+M]  = nn_proto_np
#                         d_nn_idx[cursor_out:cursor_out+M]    = nn_idx_np
#                         d_nn_dist[cursor_out:cursor_out+M]   = nn_dist_np
#                         d_topk_idx[cursor_out:cursor_out+M]  = topk_idx_np
#                         d_topk_dist[cursor_out:cursor_out+M] = topk_dist_np

#                         d_sim_d1[cursor_out:cursor_out+M]      = sim_d1_np
#                         d_sim_d1_norm[cursor_out:cursor_out+M] = sim_d1_norm_np
#                         d_recon[cursor_out:cursor_out+M]       = recon_np
#                         d_recon_norm[cursor_out:cursor_out+M]  = recon_norm_np
#                         d_score_total[cursor_out:cursor_out+M] = score_total_np

#                         cursor_out += M

#                 cursor_test += B
#                 del x_hat, s_last, proto, inputs
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()

#         print(f"[auto_focus_dump_segmented] segment {seg_id} ({a}-{b}) saved -> {out_h5_path}")

#     print("[auto_focus_dump_segmented] all segments done.")

import os
from typing import List, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def auto_focus_dump_segmented(
    model,
    test_loader,
    args,
    out_dir: str,                    # 디렉터리. 세그먼트별로 개별 h5 저장
    detect_by: str = "labels",       # "labels" | "scores_auto"
    score_mode: str = None,
    q_high: float = 0.995,
    margin_left: int = 200,
    margin_right: int = 200,
    stride: int = 50,
    topk: int = 3,
    train_size_for_abs: Optional[int] = None,
):
    """
    [개선/메모리절약 버전]
    - 세그먼트 H5에 다음을 '반드시' 저장 (기존과 동일):
        /auto_focus/attn           (N,H,I,I)
        /auto_focus/nn_proto       (N,H,I,I)
        /auto_focus/nn_idx         (N,H)
        /auto_focus/nn_dist        (N,H)
        /auto_focus/sim_d1         (N,)
        /auto_focus/sim_d1_norm    (N,)
        /auto_focus/recon          (N,)
        /auto_focus/recon_norm     (N,)
        /auto_focus/score_total    (N,)
        /auto_focus/topk_idx       (N,H,k)
        /auto_focus/topk_dist      (N,H,k)
        /auto_focus/global_idx     (N,)
        /auto_focus/test_rel_idx   (N,)
        /auto_focus/labels         (N,)
        /auto_focus/prototypes     (K,I,I)

    - 1PASS: score/label로 이상 구간 찾기 (기존과 동일)
    - 2PASS: test_loader를 "한 번만" 순회하면서,
             각 시점이 속하는 segment들에만 H5 append (stride 적용)
    """
    assert train_size_for_abs is not None, "train_size_for_abs(len(train)) 필요"
    os.makedirs(out_dir, exist_ok=True)

    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    similarity_metric = getattr(args, "similarity_metric", "cosine")
    score_mode = score_mode or getattr(args, "score_mode", "recon")

    model.eval()
    criterion = nn.MSELoss(reduction="none")

    # 정규화 통계 (compute_norm_stats에서 register_buffer로 넣었다고 가정)
    recon_avg = getattr(model, "recon_avg", torch.tensor(0.0)).detach().cpu()
    recon_std = getattr(model, "recon_std", torch.tensor(1.0)).detach().cpu()
    sim_avg   = getattr(model, "sim_avg",   torch.tensor(0.0)).detach().cpu()
    sim_std   = getattr(model, "sim_std",   torch.tensor(1.0)).detach().cpu()

    recon_avg_val = float(recon_avg.item())
    recon_std_val = float(recon_std.item())
    sim_avg_val   = float(sim_avg.item())
    sim_std_val   = float(sim_std.item())

    # --- 스코어 계산 유틸 (1PASS)
    def _compute_score(x_hat, inputs, proto, s_last):
        # recon
        rec = criterion(x_hat, inputs).mean(dim=1).mean(dim=1)  # [B]
        rec_np = rec.detach().cpu().numpy().astype(np.float32)
        recon_score = (rec_np - recon_avg_val) / (recon_std_val + 1e-12)

        if score_mode == "recon":
            return recon_score

        # sim
        if (proto is None) or ("s_logits" not in proto):
            return recon_score

        s_logits = proto["s_logits"]  # [B,H,K]
        if similarity_metric == "js":
            d = -s_logits
        else:
            d = (1.0 - s_logits).clamp_min(0.0)
        d1 = d.min(dim=-1).values.mean(dim=1)                     # [B]
        d1_np = d1.detach().cpu().numpy().astype(np.float32)
        sim_score = (d1_np - sim_avg_val) / (sim_std_val + 1e-12)

        if score_mode in ("recon+sim", "elementwise"):
            return recon_score + sim_score
        return recon_score

    # ---------------- 1PASS: score/label 수집 ----------------
    all_scores_list: List[np.ndarray] = []
    all_labels_list: List[np.ndarray] = []

    need_proto_first = (
        score_mode != "recon"
        and getattr(model, "proto_bank", None) is not None
        and getattr(model, "K", 0) > 0
    )

    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        x_hat, s_last, proto, _ = model(
            inputs, compute_proto=need_proto_first, detach_attn=True
        )
        scores = _compute_score(x_hat, inputs, proto, s_last)
        all_scores_list.append(scores.astype(np.float32))
        all_labels_list.append(labels.numpy().astype(np.int32))

        del x_hat, s_last, proto, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_scores = np.concatenate(all_scores_list, axis=0)  # [T_test]
    all_labels = np.concatenate(all_labels_list, axis=0)  # [T_test]
    del all_scores_list, all_labels_list

    T_test = all_scores.shape[0]

    # ---------------- 이상 구간 추출 ----------------
    def _ranges_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
        ranges = []
        N = mask.shape[0]
        i = 0
        while i < N:
            if mask[i] == 1:
                j = i
                while j + 1 < N and mask[j + 1] == 1:
                    j += 1
                ranges.append((i, j))
                i = j + 1
            else:
                i += 1
        return ranges

    if detect_by == "labels" and (all_labels.sum() > 0):
        base_ranges = _ranges_from_mask((all_labels > 0).astype(np.int32))
    else:
        thr = float(np.quantile(all_scores, q_high))
        base_ranges = _ranges_from_mask((all_scores >= thr).astype(np.int32))

    if len(base_ranges) == 0:
        print("[auto_focus_dump_segmented] no anomaly ranges found.")
        return

    # ---------------- 세그먼트 index 매핑 준비 ----------------
    stride = max(1, int(stride))

    # seg_id -> (a, b, left, right)
    seg_configs = []
    for seg_id, (a, b) in enumerate(base_ranges):
        left = max(0, a - int(margin_left))
        right = min(T_test - 1, b + int(margin_right))
        if left > right:
            continue
        seg_configs.append((seg_id, a, b, left, right))

    if not seg_configs:
        print("[auto_focus_dump_segmented] no valid segments after margin.")
        return

    # rel index -> 포함되는 segment id 리스트
    index_to_segs: List[List[int]] = [[] for _ in range(T_test)]
    for seg_id, _a, _b, left, right in seg_configs:
        for t in range(left, right + 1, stride):
            index_to_segs[t].append(seg_id)

    # ---------------- H5 파일들 미리 열기 + 메타/프로토 저장 ----------------
    need_proto_second = (
        getattr(model, "proto_bank", None) is not None
        and getattr(model, "K", 0) > 0
    )

    if need_proto_second:
        E = (
            model.proto_bank.prototypes()
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )  # (K,I,I)
        K, I, _ = E.shape
    else:
        E = None

    seg_handles = {}  # seg_id -> dict

    for seg_id, a, b, left, right in seg_configs:
        out_h5_path = os.path.join(out_dir, f"segment_{seg_id}_a{a}_b{b}.h5")
        f = h5py.File(out_h5_path, "w")
        g = f.create_group("auto_focus")

        if need_proto_second and E is not None:
            g.create_dataset("prototypes", data=E, compression="gzip")
            g.attrs["proto_K"] = int(E.shape[0])
            g.attrs["proto_I"] = int(E.shape[1])

        # 메타/세팅
        g.attrs["similarity_metric"] = str(similarity_metric)
        g.attrs["score_mode"] = str(score_mode)
        g.attrs["margin_left"] = int(margin_left)
        g.attrs["margin_right"] = int(margin_right)
        g.attrs["stride"] = int(stride)
        g.attrs["q_high"] = float(q_high)
        g.attrs["topk"] = int(topk)
        g.attrs["recon_avg"] = float(recon_avg_val)
        g.attrs["recon_std"] = float(recon_std_val)
        g.attrs["sim_avg"] = float(sim_avg_val)
        g.attrs["sim_std"] = float(sim_std_val)

        # 기본 인덱스/라벨 DS (길이 0, maxshape None)
        d_global = g.create_dataset(
            "global_idx", shape=(0,), maxshape=(None,), dtype="int64"
        )
        d_rel = g.create_dataset(
            "test_rel_idx", shape=(0,), maxshape=(None,), dtype="int64"
        )
        d_labels = g.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype="int8"
        )

        seg_handles[seg_id] = dict(
            path=out_h5_path,
            file=f,
            group=g,
            d_global=d_global,
            d_rel=d_rel,
            d_labels=d_labels,
            d_attn=None,
            d_nn_proto=None,
            d_nn_idx=None,
            d_nn_dist=None,
            d_topk_idx=None,
            d_topk_dist=None,
            d_sim_d1=None,
            d_sim_d1_norm=None,
            d_recon=None,
            d_recon_norm=None,
            d_score_total=None,
            cursor=0,
        )

    # ---------------- 2PASS: test_loader 한 번만 순회하며 각 segment에 append ----------------
    cursor_test = 0

    for inputs, _labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        x_hat, s_last, proto, _ = model(
            inputs, compute_proto=need_proto_second, detach_attn=True
        )
        B = inputs.size(0)

        # 이 배치에 해당하는 전체 rel index
        rel_idx_full = np.arange(cursor_test, cursor_test + B, dtype=np.int64)

        # 배치 내에서, 각 seg_id가 가져갈 sample index 모으기
        seg_to_batch_idx: dict[int, List[int]] = {}
        any_used = False
        for j, rel_idx in enumerate(rel_idx_full):
            seg_list = index_to_segs[rel_idx]
            if not seg_list:
                continue
            any_used = True
            for seg_id in seg_list:
                if seg_id not in seg_to_batch_idx:
                    seg_to_batch_idx[seg_id] = []
                seg_to_batch_idx[seg_id].append(j)

        if not any_used:
            cursor_test += B
            del x_hat, s_last, proto, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # 공통 계산 (배치 전체 기준으로 한 번만)
        rec_full = criterion(x_hat, inputs).mean(dim=1).mean(dim=1)  # [B]
        rec_full_np = rec_full.detach().cpu().numpy().astype(np.float32)

        if need_proto_second and (proto is not None) and ("s_logits" in proto):
            s_logits_full = proto["s_logits"]  # [B,H,K]
            if similarity_metric == "js":
                d_full = -s_logits_full
            else:
                d_full = (1.0 - s_logits_full).clamp_min(0.0)        # [B,H,K]

            k_use = min(int(topk), d_full.size(-1))
            vals_full, idxs_full = torch.topk(
                d_full, k=k_use, dim=-1, largest=False
            )  # [B,H,k]
            nn_idx_full = idxs_full[..., 0]                         # [B,H]
            nn_dist_full = vals_full[..., 0]                        # [B,H]
            sim_d1_full = nn_dist_full.mean(dim=1)                  # [B]
        else:
            k_use = int(topk)
            s_logits_full = None
            vals_full = idxs_full = nn_idx_full = nn_dist_full = None
            sim_d1_full = None

        # seg별로 실제 append
        for seg_id, batch_indices in seg_to_batch_idx.items():
            handle = seg_handles[seg_id]
            if len(batch_indices) == 0:
                continue

            keep_idx_local = torch.as_tensor(
                batch_indices, device=inputs.device, dtype=torch.long
            )
            M = keep_idx_local.numel()

            # --- 어텐션 선택
            s_pick = s_last.index_select(0, keep_idx_local)         # [M,H,I,I]
            _, H, I, _ = s_pick.shape
            s_np = s_pick.detach().cpu().numpy().astype(np.float32)

            # --- 재구성 오차
            rec_sel_np = rec_full_np[batch_indices]                 # [M]
            recon_np = rec_sel_np.astype(np.float32)
            recon_norm_np = (recon_np - recon_avg_val) / (recon_std_val + 1e-12)

            # --- 프로토 관련
            if need_proto_second and s_logits_full is not None:
                nn_idx_t = nn_idx_full.index_select(0, keep_idx_local)   # [M,H]
                nn_dist_t = nn_dist_full.index_select(0, keep_idx_local) # [M,H]
                sim_d1_t = sim_d1_full.index_select(0, keep_idx_local)   # [M]

                nn_idx_np = nn_idx_t.detach().cpu().numpy().astype(np.int32)
                nn_dist_np = nn_dist_t.detach().cpu().numpy().astype(np.float32)
                sim_d1_np = sim_d1_t.detach().cpu().numpy().astype(np.float32)

                sim_d1_norm_np = (sim_d1_np - sim_avg_val) / (sim_std_val + 1e-12)

                # 최근접 프로토 행렬: E[nn_idx] (벡터화)
                if E is not None:
                    nn_proto_np = E[nn_idx_np.reshape(-1)].reshape(M, H, I, I).astype(
                        np.float32
                    )
                else:
                    nn_proto_np = np.full((M, H, I, I), np.nan, dtype=np.float32)

                # top-k
                topk_idx_t = idxs_full.index_select(0, keep_idx_local)   # [M,H,k]
                topk_dist_t = vals_full.index_select(0, keep_idx_local)  # [M,H,k]
                topk_idx_np = topk_idx_t.detach().cpu().numpy().astype(np.int32)
                topk_dist_np = topk_dist_t.detach().cpu().numpy().astype(np.float32)
            else:
                nn_idx_np = -np.ones((M, H), dtype=np.int32)
                nn_dist_np = np.full((M, H), np.nan, dtype=np.float32)
                sim_d1_np = np.full((M,), np.nan, dtype=np.float32)
                sim_d1_norm_np = np.full((M,), np.nan, dtype=np.float32)
                nn_proto_np = np.full((M, H, I, I), np.nan, dtype=np.float32)
                topk_idx_np = -np.ones((M, H, k_use), dtype=np.int32)
                topk_dist_np = np.full((M, H, k_use), np.nan, dtype=np.float32)

            # --- 최종 score_total
            if score_mode == "recon":
                score_total_np = recon_norm_np.astype(np.float32)
            elif score_mode in ("recon+sim", "elementwise"):
                score_total_np = (recon_norm_np + sim_d1_norm_np).astype(np.float32)
            else:
                score_total_np = recon_norm_np.astype(np.float32)

            # --- 인덱스/라벨
            rel_idx_kept = rel_idx_full[batch_indices]                    # [M]
            abs_idx_kept = rel_idx_kept + int(train_size_for_abs)         # [M]
            labels_np = _labels.numpy().astype(np.int8)[batch_indices]    # [M]

            rel_np = rel_idx_kept.astype(np.int64)
            abs_np = abs_idx_kept.astype(np.int64)

            # --- H5 append
            cur = handle["cursor"]

            # dataset가 없으면 생성
            if handle["d_attn"] is None:
                g = handle["group"]
                handle["d_attn"] = g.create_dataset(
                    "attn",
                    shape=s_np.shape,
                    maxshape=(None,) + s_np.shape[1:],
                    dtype="float32",
                    compression="gzip",
                )
                handle["d_nn_proto"] = g.create_dataset(
                    "nn_proto",
                    shape=nn_proto_np.shape,
                    maxshape=(None,) + nn_proto_np.shape[1:],
                    dtype="float32",
                    compression="gzip",
                )
                handle["d_nn_idx"] = g.create_dataset(
                    "nn_idx",
                    shape=nn_idx_np.shape,
                    maxshape=(None,) + nn_idx_np.shape[1:],
                    dtype="int32",
                    compression="gzip",
                )
                handle["d_nn_dist"] = g.create_dataset(
                    "nn_dist",
                    shape=nn_dist_np.shape,
                    maxshape=(None,) + nn_dist_np.shape[1:],
                    dtype="float32",
                    compression="gzip",
                )
                handle["d_topk_idx"] = g.create_dataset(
                    "topk_idx",
                    shape=topk_idx_np.shape,
                    maxshape=(None,) + topk_idx_np.shape[1:],
                    dtype="int32",
                    compression="gzip",
                )
                handle["d_topk_dist"] = g.create_dataset(
                    "topk_dist",
                    shape=topk_dist_np.shape,
                    maxshape=(None,) + topk_dist_np.shape[1:],
                    dtype="float32",
                    compression="gzip",
                )
                handle["d_sim_d1"] = g.create_dataset(
                    "sim_d1",
                    shape=sim_d1_np.shape,
                    maxshape=(None,),
                    dtype="float32",
                    compression="gzip",
                )
                handle["d_sim_d1_norm"] = g.create_dataset(
                    "sim_d1_norm",
                    shape=sim_d1_norm_np.shape,
                    maxshape=(None,),
                    dtype="float32",
                    compression="gzip",
                )
                handle["d_recon"] = g.create_dataset(
                    "recon",
                    shape=recon_np.shape,
                    maxshape=(None,),
                    dtype="float32",
                    compression="gzip",
                )
                handle["d_recon_norm"] = g.create_dataset(
                    "recon_norm",
                    shape=recon_norm_np.shape,
                    maxshape=(None,),
                    dtype="float32",
                    compression="gzip",
                )
                handle["d_score_total"] = g.create_dataset(
                    "score_total",
                    shape=score_total_np.shape,
                    maxshape=(None,),
                    dtype="float32",
                    compression="gzip",
                )

                # 인덱스/라벨 dataset resize 후 채우기
                handle["d_global"].resize((M,))
                handle["d_rel"].resize((M,))
                handle["d_labels"].resize((M,))
                handle["d_global"][:] = abs_np
                handle["d_rel"][:] = rel_np
                handle["d_labels"][:] = labels_np

                handle["d_attn"][:] = s_np
                handle["d_nn_proto"][:] = nn_proto_np
                handle["d_nn_idx"][:] = nn_idx_np
                handle["d_nn_dist"][:] = nn_dist_np
                handle["d_topk_idx"][:] = topk_idx_np
                handle["d_topk_dist"][:] = topk_dist_np
                handle["d_sim_d1"][:] = sim_d1_np
                handle["d_sim_d1_norm"][:] = sim_d1_norm_np
                handle["d_recon"][:] = recon_np
                handle["d_recon_norm"][:] = recon_norm_np
                handle["d_score_total"][:] = score_total_np

                handle["cursor"] = M
            else:
                # 뒤에 붙이기
                new_len = cur + M

                handle["d_global"].resize((new_len,))
                handle["d_rel"].resize((new_len,))
                handle["d_labels"].resize((new_len,))
                handle["d_attn"].resize((new_len,) + handle["d_attn"].shape[1:])
                handle["d_nn_proto"].resize(
                    (new_len,) + handle["d_nn_proto"].shape[1:]
                )
                handle["d_nn_idx"].resize((new_len,) + handle["d_nn_idx"].shape[1:])
                handle["d_nn_dist"].resize((new_len,) + handle["d_nn_dist"].shape[1:])
                handle["d_topk_idx"].resize(
                    (new_len,) + handle["d_topk_idx"].shape[1:]
                )
                handle["d_topk_dist"].resize(
                    (new_len,) + handle["d_topk_dist"].shape[1:]
                )
                handle["d_sim_d1"].resize((new_len,))
                handle["d_sim_d1_norm"].resize((new_len,))
                handle["d_recon"].resize((new_len,))
                handle["d_recon_norm"].resize((new_len,))
                handle["d_score_total"].resize((new_len,))

                handle["d_global"][cur:new_len] = abs_np
                handle["d_rel"][cur:new_len] = rel_np
                handle["d_labels"][cur:new_len] = labels_np

                handle["d_attn"][cur:new_len] = s_np
                handle["d_nn_proto"][cur:new_len] = nn_proto_np
                handle["d_nn_idx"][cur:new_len] = nn_idx_np
                handle["d_nn_dist"][cur:new_len] = nn_dist_np
                handle["d_topk_idx"][cur:new_len] = topk_idx_np
                handle["d_topk_dist"][cur:new_len] = topk_dist_np
                handle["d_sim_d1"][cur:new_len] = sim_d1_np
                handle["d_sim_d1_norm"][cur:new_len] = sim_d1_norm_np
                handle["d_recon"][cur:new_len] = recon_np
                handle["d_recon_norm"][cur:new_len] = recon_norm_np
                handle["d_score_total"][cur:new_len] = score_total_np

                handle["cursor"] = new_len

        cursor_test += B

        del x_hat, s_last, proto, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 파일 닫기 + 로그
    for seg_id, (a, b, _, _) in zip(
        [cfg[0] for cfg in seg_configs],
        [(cfg[1], cfg[2], cfg[3], cfg[3]) for cfg in seg_configs],
    ):
        handle = seg_handles[seg_id]
        handle["file"].close()
        print(
            f"[auto_focus_dump_segmented] segment {seg_id} saved -> {handle['path']}"
        )

    print("[auto_focus_dump_segmented] all segments done.")

# --- 작은 헬퍼(0 나눗셈 방지용) ---
def sim_max(x: float, eps: float = 1e-12) -> float:
    return max(x, eps)


import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import h5py
@torch.no_grad()
def auto_focus_dump_normal_segments(
    model,
    test_loader,
    args,
    out_dir: str,                    # 정상 세그먼트 h5들이 저장될 디렉토리
    max_segments: int = 100,         # 최대 세그먼트 개수
    detect_by: str = "labels",       # "labels" | "scores_auto"
    score_mode: str = None,          # "recon" | "recon+sim" 등 (기본은 args.score_mode 또는 "recon")
    q_low: float = 0.05,             # scores_auto일 때 하위 몇 %를 정상 후보로 볼지
    margin_left: int = 200,
    margin_right: int = 200,
    stride: int = 50,
    topk: int = 3,
    train_size_for_abs: Optional[int] = None,
    min_range_len: int = 10,         # 정상 연속 구간 최소 길이
    random_seed: int = 42,
    clean_guard: int = 400,          # ★ 이상 라벨로부터 최소 거리 (index 단위)
):
    """
    '진짜 깨끗한 normal phase'만 사용해서,
    기존 auto_focus_dump_segmented 와 동일 포맷으로 h5를 저장하는 함수.

    - 먼저 normal 후보(mask_normal)를 만든 다음,
      -> anomaly 마스크(라벨 또는 score 상위)를 만들고
      -> anomaly 주변 ±clean_guard 를 모두 제외 (dilated_anom)
      -> mask_clean = mask_normal & (~dilated_anom)
    - mask_clean 에서 연속 구간만 base_ranges 로 사용.

    각 세그먼트에 대해:
      /auto_focus/attn, nn_proto, nn_idx, nn_dist, topk_idx, topk_dist
      /auto_focus/sim_d1, sim_d1_norm, recon, recon_norm, score_total
      /auto_focus/global_idx, test_rel_idx, labels
      /auto_focus/prototypes  (파일당 1회)
    를 저장한다.
    """
    assert train_size_for_abs is not None, "train_size_for_abs(len(train)) 필요"

    os.makedirs(out_dir, exist_ok=True)

    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    similarity_metric = getattr(args, "similarity_metric", "cosine")
    score_mode = score_mode or getattr(args, "score_mode", "recon")

    model.eval()
    criterion = nn.MSELoss(reduction="none")

    # --- 정규화 통계 ---
    recon_avg_t = getattr(model, "recon_avg", torch.tensor(0.0)).detach().cpu()
    recon_std_t = getattr(model, "recon_std", torch.tensor(1.0)).detach().cpu()
    sim_avg_t   = getattr(model, "sim_avg",   torch.tensor(0.0)).detach().cpu()
    sim_std_t   = getattr(model, "sim_std",   torch.tensor(1.0)).detach().cpu()

    recon_avg = float(recon_avg_t.numpy() if recon_avg_t.ndim > 0 else recon_avg_t.item())
    recon_std = float(recon_std_t.numpy() if recon_std_t.ndim > 0 else recon_std_t.item())
    sim_avg   = float(sim_avg_t.numpy()   if sim_avg_t.ndim   > 0 else sim_avg_t.item())
    sim_std   = float(sim_std_t.numpy()   if sim_std_t.ndim   > 0 else sim_std_t.item())

    need_proto = (getattr(model, "proto_bank", None) is not None) and (getattr(model, "K", 0) > 0)

    # --- 스코어 계산 유틸 (1PASS용) ---
    def _compute_score(x_hat, inputs, proto):
        rec = criterion(x_hat, inputs).mean(dim=1).mean(dim=1)  # [B]
        recon_score = ((rec.detach().cpu().numpy().astype(np.float32) - recon_avg)
                       / (recon_std + 1e-12))

        if score_mode == "recon":
            return recon_score

        if (proto is None) or ("s_logits" not in proto):
            return recon_score

        s_logits = proto["s_logits"]  # [B,H,K]
        if similarity_metric == "js":
            d = -s_logits
        else:
            d = (1.0 - s_logits).clamp_min(0.0)
        d1 = d.min(dim=-1).values.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [B]
        sim_score = (d1 - sim_avg) / (sim_std + 1e-12)

        if score_mode in ("recon+sim", "elementwise"):
            return recon_score + sim_score
        return recon_score

    # ---------------- 1PASS: score/label 수집 ----------------
    all_scores_list: List[np.ndarray] = []
    all_labels_list: List[np.ndarray] = []

    need_proto_first = (score_mode != "recon") and need_proto

    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        x_hat, s_last, proto, _ = model(inputs, compute_proto=need_proto_first, detach_attn=True)
        scores = _compute_score(x_hat, inputs, proto)
        all_scores_list.append(scores.astype(np.float32))
        all_labels_list.append(labels.numpy().astype(np.int32))

        del x_hat, s_last, proto, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_scores = np.concatenate(all_scores_list, axis=0)  # [T_test]
    all_labels = np.concatenate(all_labels_list, axis=0)  # [T_test]
    T_total = all_scores.shape[0]

    del all_scores_list, all_labels_list

    # ---------------- 정상/이상 마스크 & 클린 마스크 ----------------
    def _ranges_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
        ranges = []
        N = mask.shape[0]
        i = 0
        while i < N:
            if mask[i]:
                j = i
                while j + 1 < N and mask[j + 1]:
                    j += 1
                ranges.append((i, j))
                i = j + 1
            else:
                i += 1
        return ranges

    # (1) normal 후보
    if detect_by == "labels" and np.any(all_labels == 0):
        mask_normal = (all_labels == 0)
    else:
        thr_low = float(np.quantile(all_scores, q_low))
        mask_normal = (all_scores <= thr_low)

    # (2) anomaly 마스크
    if detect_by == "labels" and np.any(all_labels > 0):
        mask_anom = (all_labels > 0)
    else:
        # label 기준이 없으면 score 상위 q_high 를 anomaly 로 사용
        q_high = 0.995
        thr_high = float(np.quantile(all_scores, q_high))
        mask_anom = (all_scores >= thr_high)

    # (3) anomaly 주변 ±clean_guard 모두 제거
    dilated_anom = np.zeros_like(mask_anom, dtype=bool)
    idx_anom = np.where(mask_anom)[0]
    for t in idx_anom:
        a = max(0, t - clean_guard)
        b = min(T_total - 1, t + clean_guard)
        dilated_anom[a:b+1] = True

    # (4) 최종 "깨끗한" normal 마스크
    mask_clean = mask_normal & (~dilated_anom)

    normal_ranges = _ranges_from_mask(mask_clean)
    normal_ranges = [(a, b) for (a, b) in normal_ranges if (b - a + 1) >= min_range_len]

    rng = np.random.default_rng(random_seed)

    if len(normal_ranges) == 0:
        print("[auto_focus_dump_normal_segments] no CLEAN normal ranges found.")
        return

    # 너무 많으면 max_segments 개만 랜덤 샘플
    if len(normal_ranges) > max_segments:
        chosen = rng.choice(len(normal_ranges), size=max_segments, replace=False)
        chosen = sorted(chosen.tolist())
        base_ranges = [normal_ranges[i] for i in chosen]
    else:
        base_ranges = normal_ranges

    if len(base_ranges) == 0:
        print("[auto_focus_dump_normal_segments] no normal ranges selected.")
        return

    print(f"[auto_focus_dump_normal_segments] selected {len(base_ranges)} CLEAN normal ranges.")

    # ---------------- 세그먼트별 focus index 및 인덱스→세그먼트 매핑 ----------------
    segments = []
    for seg_id, (a, b) in enumerate(base_ranges):
        left = max(0, a - int(margin_left))
        right = min(T_total - 1, b + int(margin_right))
        if right < left:
            continue
        idx_list = list(range(left, right + 1, max(1, int(stride))))
        if not idx_list:
            continue
        segments.append({
            "id": seg_id,
            "a": a,
            "b": b,
            "left": left,
            "right": right,
            "indices": idx_list,
        })

    if not segments:
        print("[auto_focus_dump_normal_segments] no valid segments after margin/stride.")
        return

    seg_count = len(segments)
    print(f"[auto_focus_dump_normal_segments] final segments: {seg_count}")

    # index -> segment id 리스트 (겹치는 경우도 허용)
    idx_to_segs: List[List[int]] = [list() for _ in range(T_total)]
    for seg in segments:
        sid = seg["id"]
        for t in seg["indices"]:
            idx_to_segs[t].append(sid)

    # ---------------- 프로토타입 E 가져오기 ----------------
    if need_proto:
        E = model.proto_bank.prototypes().detach().cpu().numpy().astype("float32")  # (K,I,I)
        K = E.shape[0]
    else:
        E = None
        K = 0  # (실제로 아래에서 쓰지는 않지만 형식상 남겨둠)

    # ---------------- 세그먼트별 버퍼 준비 ----------------
    buffers = []
    for _ in segments:
        buffers.append({
            "global_idx": [],
            "rel_idx": [],
            "labels": [],
            "attn": [],
            "nn_proto": [],
            "nn_idx": [],
            "nn_dist": [],
            "topk_idx": [],
            "topk_dist": [],
            "sim_d1": [],
            "sim_d1_norm": [],
            "recon": [],
            "recon_norm": [],
            "score_total": [],
        })

    # ---------------- 2PASS: 필요한 index들만 다시 계산해서 버퍼에 쌓기 ----------------
    need_proto_second = need_proto
    cursor_test = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels_np = labels.numpy().astype(np.int8)

        x_hat, s_last, proto, _ = model(inputs, compute_proto=need_proto_second, detach_attn=True)
        B = inputs.size(0)

        rel_idx_full = np.arange(cursor_test, cursor_test + B, dtype=np.int64)

        # 재구성 오차
        rec = criterion(x_hat, inputs).mean(dim=1).mean(dim=1)  # [B]
        rec_np = rec.detach().cpu().numpy().astype(np.float32)
        rec_norm_np = (rec_np - recon_avg) / (recon_std + 1e-12)

        # 프로토타입 관련
        if need_proto_second and (proto is not None) and ("s_logits" in proto):
            s_logits = proto["s_logits"]  # [B,H,K]
            if similarity_metric == "js":
                d = -s_logits
            else:
                d = (1.0 - s_logits).clamp_min(0.0)

            k_use = min(int(topk), d.size(-1))
            vals, idxs = torch.topk(d, k=k_use, dim=-1, largest=False)  # [B,H,k]
            nn_idx = idxs[..., 0]   # [B,H]
            nn_dist = vals[..., 0]  # [B,H]

            sim_d1 = nn_dist.mean(dim=1)  # [B]
            sim_d1_np = sim_d1.detach().cpu().numpy().astype(np.float32)
            sim_d1_norm_np = (sim_d1_np - sim_avg) / (sim_std + 1e-12)

            nn_idx_np = nn_idx.detach().cpu().numpy().astype(np.int32)
            nn_dist_np = nn_dist.detach().cpu().numpy().astype(np.float32)
            topk_idx_np = idxs.detach().cpu().numpy().astype(np.int32)
            topk_dist_np = vals.detach().cpu().numpy().astype(np.float32)

            s_np = s_last.detach().cpu().numpy().astype(np.float32)  # [B,H,I,I]
            _, H, I, _ = s_np.shape

            if E is not None:
                nn_proto_np = np.zeros_like(s_np, dtype=np.float32)
                for b in range(B):
                    nn_proto_np[b] = E[nn_idx_np[b]]  # (H,I,I)
            else:
                nn_proto_np = np.full_like(s_np, np.nan, dtype=np.float32)
        else:
            # 프로토타입 사용 안 하는 경우
            s_np = s_last.detach().cpu().numpy().astype(np.float32)  # [B,H,I,I]
            _, H, I, _ = s_np.shape
            k_use = int(topk)
            nn_idx_np   = -np.ones((B, H), dtype=np.int32)
            nn_dist_np  = np.full((B, H), np.nan, dtype=np.float32)
            sim_d1_np       = np.full((B,), np.nan, dtype=np.float32)
            sim_d1_norm_np  = np.full((B,), np.nan, dtype=np.float32)
            nn_proto_np = np.full((B, H, I, I), np.nan, dtype=np.float32)
            topk_idx_np  = -np.ones((B, H, k_use), dtype=np.int32)
            topk_dist_np = np.full((B, H, k_use), np.nan, dtype=np.float32)

        # 배치 내 각 time index에 대해, 해당되는 세그먼트 버퍼에 push
        for i in range(B):
            t = int(rel_idx_full[i])
            seg_ids = idx_to_segs[t]
            if not seg_ids:
                continue

            g_idx = t + int(train_size_for_abs)
            lab = labels_np[i]
            att_i = s_np[i]
            nn_p_i = nn_proto_np[i]
            nn_idx_i = nn_idx_np[i]
            nn_dist_i = nn_dist_np[i]
            topk_idx_i = topk_idx_np[i]
            topk_dist_i = topk_dist_np[i]
            sim_d1_i = sim_d1_np[i]
            sim_d1_norm_i = sim_d1_norm_np[i]
            rec_i = rec_np[i]
            rec_norm_i = rec_norm_np[i]

            if score_mode == "recon":
                score_i = rec_norm_i
            elif score_mode in ("recon+sim", "elementwise"):
                score_i = rec_norm_i + sim_d1_norm_i
            else:
                score_i = rec_norm_i

            for sid in seg_ids:
                buf = buffers[sid]
                buf["global_idx"].append(g_idx)
                buf["rel_idx"].append(t)
                buf["labels"].append(lab)
                buf["attn"].append(att_i)
                buf["nn_proto"].append(nn_p_i)
                buf["nn_idx"].append(nn_idx_i)
                buf["nn_dist"].append(nn_dist_i)
                buf["topk_idx"].append(topk_idx_i)
                buf["topk_dist"].append(topk_dist_i)
                buf["sim_d1"].append(sim_d1_i)
                buf["sim_d1_norm"].append(sim_d1_norm_i)
                buf["recon"].append(rec_i)
                buf["recon_norm"].append(rec_norm_i)
                buf["score_total"].append(score_i)

        cursor_test += B

        del x_hat, s_last, proto, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- 세그먼트별로 H5 저장 ----------------
    for seg, buf in zip(segments, buffers):
        if len(buf["rel_idx"]) == 0:
            continue

        seg_id = seg["id"]
        a, b = seg["a"], seg["b"]

        out_h5_path = os.path.join(out_dir, f"normal_segment_{seg_id}_a{a}_b{b}.h5")
        print(f"[auto_focus_dump_normal_segments] saving seg {seg_id} -> {out_h5_path}")

        with h5py.File(out_h5_path, "w") as f:
            g = f.create_group("auto_focus")

            # 프로토타입 뱅크
            if E is not None:
                g.create_dataset("prototypes", data=E, compression="gzip")
                g.attrs["proto_K"] = int(E.shape[0])
                g.attrs["proto_I"] = int(E.shape[1])

            # 메타 정보
            g.attrs["similarity_metric"] = str(similarity_metric)
            g.attrs["score_mode"] = str(score_mode)
            g.attrs["margin_left"] = int(margin_left)
            g.attrs["margin_right"] = int(margin_right)
            g.attrs["stride"] = int(stride)
            g.attrs["q_low"] = float(q_low)
            g.attrs["topk"] = int(topk)
            g.attrs["recon_avg"] = float(recon_avg)
            g.attrs["recon_std"] = float(recon_std)
            g.attrs["sim_avg"] = float(sim_avg)
            g.attrs["sim_std"] = float(sim_std)
            g.attrs["clean_guard"] = int(clean_guard)

            # 리스트 -> numpy
            def _stack(x, dtype=None):
                arr = np.stack(x, axis=0)
                return arr.astype(dtype) if dtype is not None else arr

            g.create_dataset("global_idx",   data=np.array(buf["global_idx"], dtype=np.int64))
            g.create_dataset("test_rel_idx", data=np.array(buf["rel_idx"], dtype=np.int64))
            g.create_dataset("labels",       data=np.array(buf["labels"], dtype=np.int8))

            g.create_dataset("attn",      data=_stack(buf["attn"], dtype=np.float32), compression="gzip")
            g.create_dataset("nn_proto",  data=_stack(buf["nn_proto"], dtype=np.float32), compression="gzip")
            g.create_dataset("nn_idx",    data=_stack(buf["nn_idx"], dtype=np.int32),   compression="gzip")
            g.create_dataset("nn_dist",   data=_stack(buf["nn_dist"], dtype=np.float32), compression="gzip")
            g.create_dataset("topk_idx",  data=_stack(buf["topk_idx"], dtype=np.int32),   compression="gzip")
            g.create_dataset("topk_dist", data=_stack(buf["topk_dist"], dtype=np.float32), compression="gzip")

            g.create_dataset("sim_d1",       data=np.array(buf["sim_d1"], dtype=np.float32),       compression="gzip")
            g.create_dataset("sim_d1_norm",  data=np.array(buf["sim_d1_norm"], dtype=np.float32),  compression="gzip")
            g.create_dataset("recon",        data=np.array(buf["recon"], dtype=np.float32),        compression="gzip")
            g.create_dataset("recon_norm",   data=np.array(buf["recon_norm"], dtype=np.float32),   compression="gzip")
            g.create_dataset("score_total",  data=np.array(buf["score_total"], dtype=np.float32),  compression="gzip")

    print("[auto_focus_dump_normal_segments] all normal segments done.")
