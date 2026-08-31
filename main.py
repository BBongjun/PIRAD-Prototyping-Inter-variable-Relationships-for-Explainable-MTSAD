# -*- coding: utf-8 -*-
import os
import argparse
import torch
import random
import numpy as np
import yaml

from utils.logging_utils import set_seed, log
from utils.yaml_utils import parse_args

# 데이터 모듈
from data_module.SWaT import SWaTDataModule
from data_module.SMD import SMDDataModule
from data_module.PSM import PSMDataModule
from data_module.SMAP_MSL import NASADataModule

# 새 모델/학습/평가
from model.pirad import SARADProto      # ← 새로 만든 파일/클래스
from train import train                       # ← 앞서 만들어 준 2-stage train()
# from eval import compute_norm_stats, evaluate, evaluate_and_save # ← 엔트로피+JS 기반 eval 유틸
from eval import compute_norm_stats, evaluate, evaluate_and_save_proto
from eval import auto_focus_dump_segmented, auto_focus_dump_normal_segments  # 파일 상단 import에 추가
from datetime import datetime  # ← 추가


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # 기본 디바이스/시드
    if not hasattr(args, "device"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 설정 저장
    config_path = os.path.join(args.log_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f)

    log(f"Starting experiment with config: {args}")

    # --------------------------- Data ---------------------------
    if args.dataset == 'Swat':
        data_module = SWaTDataModule(
            data_dir=args.data_dir,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            use_scaler=args.use_scaler
        )
    elif args.dataset == 'SMD':
        data_module = SMDDataModule(
            machine_id=args.machine_id,
            data_dir=args.data_dir,
            input_size=args.input_size,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_scaler=args.use_scaler
        )
    elif args.dataset == 'PSM':
        data_module = PSMDataModule(
            data_dir=args.data_dir,
            input_size=args.input_size,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_scaler=args.use_scaler,
            pin_memory=True
        )
    elif args.dataset == 'SMAP' or args.dataset == 'MSL':
        data_module = NASADataModule(
            dataset=args.dataset,
            data_dir=args.data_dir,
            # input_size=args.input_size,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_scaler=args.use_scaler,
            pin_memory=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader   = data_module.val_dataloader()
    test_loader  = data_module.test_dataloader()

    # 데이터 모듈에서 input_size 확정(없으면 args.input_size 유지)
    data_input_size = getattr(data_module, "input_size", None)
    if data_input_size is not None:
        args.input_size = data_input_size

    print("Data loaded successfully...")

    # --------------------------- Model ---------------------------
    # 하이퍼 기본값(없으면 안전 기본)
    num_prototypes      = getattr(args, "num_prototypes", 30)          # K
    proto_temperature   = getattr(args, "proto_temperature", 1.0)     # 행-softmax의 tau
    softmax_temperature = getattr(args, "softmax_temperature", 0.1)  # s→p 변환용 알파(학습 X)

    model = SARADProto(
        input_size=args.input_size,
        window_size=args.window_size,
        model_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        is_diagonal_masked=getattr(args, "is_diagnoal_masked", False),
        num_prototypes=num_prototypes,
        proto_temperature=proto_temperature,
        softmax_temperature=args.softmax_temperature,
        similarity_metric=args.similarity_metric
    )
    print("SARADProto: Model loaded successfully...")
    print(f"SARADProto: Model loaded successfully with similarity metric: '{args.similarity_metric}'")

    # --------------------------- Train ---------------------------
    model = train(model, train_loader, val_loader, args)
    model = model.to(args.device).eval()

    # --------------------------- Norm stats (val) ---------------------------
    compute_norm_stats(model, val_loader, args)
    
    # --------------------------- Evaluate (test) ---------------------------
    

    if args.save and args.seed==42:        
        model_name = getattr(args, "model", None) or model.__class__.__name__
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # [수정] 저장 파일명 변경
        save_name = f"{model_name}_finch_cosine"
        
        scores_dir = os.path.join(
            "runs", "scores",
            args.dataset,
            getattr(args, "machine_id", "NA"),
            args.model,
            f"seed{args.seed}"
        )
    
        # metrics, all_diag = evaluate_and_save_proto(
        #     model,
        #     test_loader,
        #     args,
        #     save_dir=scores_dir,
        #     save_name=save_name, # 변경된 이름 사용
        #     save_format="h5",
        #     save_attention=True,
        #     attn_stride=10000,
        #     norm_stats=None
        # )
        
        if args.auto_focus:
            len_train = len(train_loader.dataset)
            # # 저장될 h5 파일 경로 구성 (scores 파일과 나란히)
            # auto_h5 = os.path.join(scores_dir, f"{save_name}_auto_focus.h5")
            # auto_focus_dump_segmented(
            #     model=model,
            #     test_loader=test_loader,
            #     args=args,
            #     out_dir=auto_h5,
            #     detect_by='labels',      # "labels" or "scores_auto"
            #     score_mode=getattr(args, "score_mode", "recon+sim"),  # "recon", "sim", "recon+sim"
            #     margin_left=200,
            #     margin_right=200,
            #     stride=1,
            #     topk=3,
            #     train_size_for_abs=len_train
            # )
            # normal_out_dir = "runs/scores/Swat/NA/SARADProto/seed42/normal_segments_ver2"
            #             # 정상 구간
            # auto_focus_dump_normal_segments(
            #     model,
            #     test_loader,
            #     args,
            #     out_dir="runs/scores/Swat/NA/SARADProto/seed42/normal_segments_clean",
            #     max_segments=100,
            #     detect_by="labels",
            #     margin_left=200,
            #     margin_right=200,
            #     stride=1,            # time index 촘촘하게
            #     topk=3,
            #     train_size_for_abs=len_train,   # 네가 쓰던 값
            #     min_range_len=50,
            #     clean_guard=500,     # 공격으로부터 500 step 이상 떨어진 normal phase만
            # )
    else:
        evaluate(model, test_loader, args)


if __name__ == "__main__":
    main()
