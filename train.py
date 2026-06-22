# -*- coding: utf-8 -*-
from pyexpat import model
import os, math, random, copy
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
# from sklearn.cluster import KMeans   # ← 라이브러리 사용
from finch import FINCH              # [추가] FINCH 임포트
import time

def _collapse_BK(t: torch.Tensor) -> torch.Tensor:
    """
    (..., K) -> [N, K] 로 평탄화 (통계 계산용).
    """
    if t is None:
        return None
    if t.dim() == 1:
        t = t.unsqueeze(0)
    K = t.shape[-1]
    return t.reshape(-1, K)

def _save_memory_snapshot(
    model,
    save_root: str,
    epoch: int,
    proto_stats: Optional[Dict[str, Any]] = None,  # ← 여기!
    ):
    """
    proto_bank과 평가에 쓰이는 버퍼(recon_avg 등)를 하나의 .pt로 저장.
    """
    os.makedirs(save_root, exist_ok=True)
    snap = {
        'epoch': int(epoch),
        'K': int(getattr(model, 'K', 0)),
        'buffers': {}
    }

    # 프로토타입(메모리) 저장
    if hasattr(model, 'proto_bank') and (model.proto_bank is not None):
        snap['proto_bank'] = model.proto_bank.prototypes().detach().float().cpu()

    # 평가 정규화에 쓰이는 버퍼들 있으면 같이 저장
    for name in ['recon_avg','recon_std','detec_avg','detec_std','rdiag_avg','rdiag_std','ddiag_avg','ddiag_std']:
        if hasattr(model, name):
            buf = getattr(model, name)
            if torch.is_tensor(buf):
                snap['buffers'][name] = buf.detach().float().cpu()

    # 에폭 통계(선택)
    if proto_stats is not None:
        snap['proto_stats'] = proto_stats

    out_path = os.path.join(save_root, f'epoch_{epoch:03d}.pt')
    torch.save(snap, out_path)
    try:
        from utils.logging_utils import log
        log(f"[memory_debug] snapshot saved → {out_path}")
    except Exception:
        print(f"[memory_debug] snapshot saved → {out_path}")


# ---------------------------------------------------------------------
# 유틸: 마지막 레이어 어텐션 수집 (전처리 없음, cosine용 L2만)
# ---------------------------------------------------------------------
@torch.no_grad()
def _sample_attn_vectors(model, loader, device, sample_ratio=0.10, max_samples: Optional[int]=None):
    """
    마지막 레이어 어텐션 s_last ∈ [B,H,I,I] 수집 후 벡터화.
    cosine k-means를 위해 L2 정규화만 수행.
    return: X ∈ [M, I*I], I (변수 수)
    """
    model.eval()
    vecs = []
    I = None

    for x, _ in tqdm(loader, desc="[Stage-1→k-means] collect attn", leave=False):
        x = x.to(device)
        _, s_last, _, _ = model(x, compute_proto=False)  # s_last: [B,H,I,I]
        B, H, I, _ = s_last.shape
        v = s_last.reshape(B * H, -1)                         # [BH, I*I]
        # v = v / (v.norm(p=2, dim=-1, keepdim=True) + 1e-12)   # L2 normalize (cosine용)
        vecs.append(v.cpu())

    X = torch.cat(vecs, dim=0)                                # [N, D]
    target = math.ceil(X.size(0) * sample_ratio) if max_samples is None else min(max_samples, math.ceil(X.size(0) * sample_ratio))
    idx = torch.randperm(X.size(0))[:target]
    X = X[idx]                                                # [M, D]
    return X, I

# ---------------------------------------------------------------------
# 라이브러리 k-means (cosine 등가: 입력 L2 후 보통 KMeans로)
# ---------------------------------------------------------------------
# [추가] FINCH 결과로부터 실제 중심점을 계산하는 함수
@torch.no_grad()
def _calculate_centroids(data_vectors: torch.Tensor, cluster_labels: np.ndarray) -> torch.Tensor:
    """
    데이터와 클러스터 라벨을 기반으로 각 클러스터의 실제 중심점을 계산합니다.
    """
    unique_labels = sorted(np.unique(cluster_labels))
    centroids = []
    for label in unique_labels:
        # 현재 클러스터 라벨에 해당하는 모든 데이터 벡터들의 평균을 계산합니다.
        points_in_cluster = data_vectors[cluster_labels == label]
        centroid = torch.mean(points_in_cluster, dim=0)
        centroids.append(centroid)
    return torch.stack(centroids, dim=0)

@torch.no_grad()
def calculate_cosine_centroids_flat(
    data_vectors: torch.Tensor,         # [N, D] (D = I*I; attention matrix flatten)
    cluster_labels: np.ndarray,         # (N,)
    eps: float = 1e-12,
    I: Optional[int] = None,               # enforce_row_stochastic=True라면 필요
) -> torch.Tensor:
    """
    Cosine 거리(= spherical k-means) 기준의 클러스터 중심 (flatten 형태로 반환).
    절차:
      1) 각 샘플 L2 정규화 (단위벡터)
      2) 클러스터별 평균 벡터
      3) 평균 벡터도 L2 정규화 (방향만 유지)

    enforce_row_stochastic=True 이면, [K,I,I]로 복원해서 행합=1이 되도록 정규화(옵션: 대각 마스킹) 후,
    다시 flatten 해서 반환.
    """
    x = data_vectors  # [N, D]
    device = x.device

    # 1) 각 샘플 L2 정규화
    x = x / (x.norm(dim=-1, keepdim=True) + eps)

    # 라벨을 텐서로
    labels = torch.as_tensor(cluster_labels, device=device)
    uniq, inv = torch.unique(labels, return_inverse=True)  # uniq: [K'], inv: [N]
    K = uniq.numel()
    D = x.size(1)

    # 2) 클러스터별 합
    sums = torch.zeros(K, D, device=device)
    sums.index_add_(0, inv, x)  # 각 클러스터에 해당 샘플 단위벡터 합산

    # 클러스터별 개수
    counts = torch.bincount(inv, minlength=K).float().clamp_min_(1.0).unsqueeze(1)  # [K,1]

    # 평균
    means = sums / counts  # [K, D]

    # 3) 평균도 L2 정규화 (방향 벡터)
    centroids = means / (means.norm(dim=-1, keepdim=True) + eps)  # [K, D]

    C = centroids.view(K, I, I)  # [K, I, I]

    # 행합 1로 재정규화 (분모 안전 처리)
    row_sum = C.sum(dim=-1, keepdim=True).clamp_min(eps)  # [K, I, 1]
    C = C / row_sum

    centroids = C.view(K, D)

    return centroids  # [K, D]

def _calculate_medoids(data_vectors: torch.Tensor,
                       cluster_labels: np.ndarray,
                       distance: str = 'cosine') -> torch.Tensor:
    """
    각 클러스터의 medoid(군집 내 총거리 합이 최소인 원본 샘플)를 반환.
    - data_vectors: [N, D], (이미 L2 정규화 가정)
    - cluster_labels: np.ndarray, shape [N]
    - distance: 'cosine' | 'euclidean'
    return: [K, D] (각 medoid 벡터)
    """
    device = data_vectors.device
    unique_labels = sorted(np.unique(cluster_labels).tolist())
    medoids = []

    for lab in unique_labels:
        idx_np = np.where(cluster_labels == lab)[0]
        idx = torch.from_numpy(idx_np).to(device=device, dtype=torch.long)
        pts = data_vectors.index_select(0, idx)  # [m, D]
        if pts.size(0) == 1:
            medoids.append(pts[0])
            continue

        # if distance == 'cosine':
        #     # L2 정규화된 벡터 가정 → 코사인 거리 = 1 - cos(sim)
        #     sim = pts @ pts.T                       # [m, m]
        #     dist = (1.0 - sim.clamp(-1.0, 1.0))     # [m, m]
        if distance == 'cosine':
            pts = F.normalize(pts, p=2, dim=-1)        # 여기서만 단위벡터화
            sim = pts @ pts.T                               # [m, m]
            dist = (1.0 - sim.clamp(-1.0, 1.0))
        else:
            dist = torch.cdist(pts, pts, p=2)       # [m, m]

        sums = dist.sum(dim=1)                      # [m]
        medoids.append(pts[sums.argmin()])

    return torch.stack(medoids, dim=0)              # [K, D]
# ---------------------------------------------------------------------
# 손실: 엔트로피(샘플↓), 배치평균 균등화(↑)
# ---------------------------------------------------------------------
def _entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=dim)

def _equip_loss(p_batch: torch.Tensor) -> torch.Tensor:
    return -_entropy(p_batch, dim=-1).mean()  # maximize H == minimize -H

# ---------------------------------------------------------------------
# 메인 train (Stage-1 → k-means → Stage-2)
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
from torch import optim
from tqdm import tqdm

from utils.logging_utils import log

def train(model, train_loader, val_loader, args):
    """
    [단일 스테이지] Reconstruction loss만으로 주어진 에폭 학습.
    - 에폭마다 (옵션) 검증 → best 모델 보관
    - 학습 종료 후: best 모델로 attention 벡터 샘플링 → k-means → 프로토타입 초기화
    - (유지) 메모리 스냅샷 저장 기능
    - 최종: 프로토타입이 초기화된 best 모델 반환
    """
    import os, copy, torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm

    device = args.device
    model.to(device)

    # --- 하이퍼/경로 ---
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    epochs    = getattr(args, "epochs_stage1", getattr(args, "epochs", 3))  # 기존 키와 호환
    val_itv   = getattr(args, "val_interval", 1)
    # [수정] K-Means 파라미터 대신 FINCH 파라미터 사용
    finch_partition_level = getattr(args, "finch_partition_level", -3)
    km_ratio  = getattr(args, "clustering_sample_ratio", 0.10)

    os.makedirs(args.log_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.log_dir, f"{args.dataset}_seed_{args.seed}_best.pt")
    best_model = copy.deepcopy(model)

    mem_debug = bool(getattr(args, 'memory_debug', False))
    mem_dir_root = getattr(args, 'memory_debug_dir', None) or os.path.join(args.log_dir, 'memory_debug', 'stage1')


    # ==================== 단일 스테이지: reconstruction only ====================
    log(f"[Train] epochs={epochs} | objective=L_recon")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"[TR {epoch+1}/{epochs}] Train", leave=False)
        for step, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)

            optimizer.zero_grad()
            # proto 계산/분기 모두 비활성화
            x_hat, _, _, _ = model(inputs, compute_proto=False)
            rec_loss = criterion(x_hat, inputs).mean()
            loss = rec_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'rec': f"{rec_loss.item():.4f}"})

        avg_loss = total_loss / max(1, len(train_loader))
        log(f"[TR {epoch+1}/{epochs}] Train Loss: {avg_loss:.4f}")

        # ---- (옵션) 검증 ----
        if (epoch + 1) % val_itv == 0 and (val_loader is not None):
            model.eval()
            val_loss_total = 0.0
            val_bar = tqdm(val_loader, desc=f"[TR {epoch+1}] Val", leave=False)
            with torch.no_grad():
                for step, (inputs, _) in enumerate(val_bar):
                    inputs = inputs.to(device)
                    x_hat, _, _, _ = model(inputs, compute_proto=False)
                    rec_loss = criterion(x_hat, inputs).mean()
                    loss = rec_loss
                    val_loss_total += loss.item()
                    if step % 50 == 0:
                        val_bar.set_postfix({'val_loss': f"{loss.item():.4f}", 'rec': f"{rec_loss.item():.4f}"})

            avg_val_loss = val_loss_total / max(1, len(val_loader))
            log(f"[TR {epoch+1}/{epochs}] Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_model = copy.deepcopy(model)
                best_val_loss = avg_val_loss
                log(f"[TR {epoch+1}] Best model updated (val={best_val_loss:.4f})")

        # ---- 에폭 종료: 메모리 스냅샷 저장(유지) ----
        # if mem_debug:
        #     _save_memory_snapshot(
        #         model=model,
        #         save_root=mem_dir_root,
        #         epoch=epoch + 1,
        #         proto_stats=None  # proto 사용 안 함
        #     )

        scheduler.step()
        log(f"[TR {epoch+1}] LR: {scheduler.get_last_lr()[0]:.6f}")
    # ==================== [수정] 학습 종료 후: best 모델로 FINCH 초기화 ====================
    # model = best_model
    model.to(device)

    if (getattr(model, 'proto_bank', None) is not None):
        try:
            X, I = _sample_attn_vectors(model, train_loader, device, sample_ratio=km_ratio)
            log(f"[FINCH] Running FINCH clustering on {X.shape[0]} samples...")

            # ★ 임계값(논문 스타일) 설정: 기본 50 (K < target_K 되는 첫 partition 선택)
            target_K = int(getattr(args, "target_K", 50))
            why = getattr(args, "finch_why", "first_below")  # first_below | closest
            start = time.time()

            # ---- FINCH 전체 파티션 실행 (req_clust 사용하지 않음) ----
            c, num_clust, _ = FINCH(X.numpy(), verbose=True)  # c: (N, L), num_clust: (L,)
            
            print("time for conducting Kmeans Clustering :", time.time() - start)
            print('K means clustering is done!!!')
            
            ks = np.asarray(num_clust)

            # ---- 임계값 기반 파티션 선택: first_below -> 없으면 closest ----
            below_idx = np.where(ks < target_K)[0]
            if args.finch_why == 'first_below' and len(below_idx) > 0:
                part_idx = int(below_idx[0])                       # 처음으로 K<thresh가 되는 partition
                why = "first_below"
            else:
                part_idx = int(np.argmin(np.abs(ks - target_K)))   # 가장 가까운 K
                why = "closest"

            labels = c[:, part_idx]
            new_K = int(ks[part_idx])
            log(f"[FINCH] Selected partition {part_idx} with K={new_K} (target={target_K}, mode={why})")

            # --- ProtoBank 크기 조정 ---
            if model.K != new_K:
                log(f"[ProtoBank] Resizing ProtoBank from K={model.K} to K={new_K}")
                model.K = new_K
                from model.sarad_mem import ProtoBank
                model.proto_bank = ProtoBank(
                    num_prototypes=new_K,
                    input_size=model.I,
                    temperature=getattr(model.proto_bank, 'tau', 1.0)
                ).to(device)

            # --- medoid로 프로토타입 초기화 ---
            #   X: [N, D] (torch.Tensor), labels: (N,) (np.ndarray)
            if args.prototype_type == 'medoid':
                med = _calculate_medoids(X.to(device), labels, distance='cosine')  # [K, D]
                medoids = med.view(new_K, I, I)                                    # [K, I, I]
                model.init_prototypes_from_centroids(medoids)
            elif args.prototype_type == 'centroid':
                cen = _calculate_centroids(X.to(device), labels)                   # [K, D] euclidean 기준
                # cen = calculate_cosine_centroids_flat(X.to(device), labels, I=I)  # [K, D] cosine 기준
                centroids = cen.view(new_K, I, I)                                  # [K, I, I]
                model.init_prototypes_from_centroids(centroids)
            log(f"[FINCH] Initialized {new_K} prototypes (threshold={target_K}) using MEDOIDS.")

            # --- [추가] 스냅샷: FINCH 초기화 직후 상태 저장 ---
            if mem_debug:
                _save_memory_snapshot(
                    model=model,
                    save_root=os.path.join(args.log_dir, 'memory_debug', 'post_finch'),
                    epoch=0,
                    proto_stats={
                        'finch': {
                            'target_K': target_K,
                            'selected_partition': int(part_idx),
                            'selected_K': int(new_K),
                            'num_clust': [int(k) for k in ks.tolist()],
                            'mode': why,
                        }
                    }
                )

        except Exception as e:
            log(f"[Warn] FINCH init skipped due to error: {e}")
    else:
        log("[Info] model has no proto_bank → skip FINCH init.")

    # # ==================== 저장 및 반환 ====================
    try:
        # torch.save(model.state_dict(), best_model_path)  # 필요시 활성화
        log(f"[Done] best model saved to: {best_model_path}")
    except Exception as e:
        log(f"[Warn] save failed: {e}")

    return model
    # # ==================== [수정] 학습 종료 후: best 모델로 FINCH 초기화 ====================
    # model = best_model
    # model.to(device)

    # if (getattr(model, 'proto_bank', None) is not None):
    #     try:
    #         X, I = _sample_attn_vectors(model, train_loader, device, sample_ratio=km_ratio)
    #         log(f"[FINCH] Running FINCH clustering on {X.shape[0]} samples...")

    #         # ★ 목표 K 설정 (예: 50). args.target_K 없으면 50 사용
    #         target_K = int(getattr(args, "target_K", 50))

    #         # ★ req_clust로 먼저 시도
    #         c, num_clust, req_c = FINCH(X.numpy(), req_clust=target_K, verbose=True)  # c: [N, L], num_clust: [L]

    #         if req_c is not None:
    #             # 요청 K에 해당(또는 가장 먼저 K<=target_K가 된) 라벨
    #             labels = req_c
    #             new_K = int(np.unique(labels).size)
    #             log(f"[FINCH] req_clust satisfied (or closest below): K={new_K}")
    #         else:
    #             # ★ 정확 매칭 실패 → num_clust 중 target_K에 가장 가까운 파티션 선택
    #             ks = np.array(num_clust)
    #             idx = int(np.argmin(np.abs(ks - target_K)))
    #             labels = c[:, idx]
    #             new_K = int(ks[idx])
    #             log(f"[FINCH] req_clust not satisfied → fallback to partition {idx} with K={new_K} (target={target_K})")

    #         # --- ProtoBank 크기 조정 ---
    #         if model.K != new_K:
    #             log(f"[ProtoBank] Resizing ProtoBank from K={model.K} to K={new_K}")
    #             model.K = new_K
    #             from model.sarad_mem import ProtoBank
    #             model.proto_bank = ProtoBank(
    #                 num_prototypes=new_K,
    #                 input_size=model.I,
    #                 temperature=getattr(model.proto_bank, 'tau', 1.0)
    #             ).to(device)

    #         # --- medoid로 프로토타입 초기화 ---
    #         med = _calculate_medoids(X.to(device), labels, distance='cosine')  # [K, D]
    #         medoids = med.view(new_K, I, I)                                    # [K, I, I]
    #         model.init_prototypes_from_centroids(medoids)
    #         log(f"[FINCH] Initialized {new_K} prototypes (target={target_K}) using MEDOIDS.")

    #     except Exception as e:
    #         log(f"[Warn] FINCH init skipped due to error: {e}")
    # else:
    #     log("[Info] model has no proto_bank → skip FINCH init.")

    # return model