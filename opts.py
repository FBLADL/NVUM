import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="CXR with MM")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--total_epochs", default=30, type=int)

    parser.add_argument("--train_data", default="NIH", type=str)
    parser.add_argument("--train_root_dir", default="/run/media/Data/chestxray/")
    parser.add_argument("--openi_root_dir", default="/run/media/Data/open-i/")
    parser.add_argument("--pc_root_dir", default="/run/media/Data/PADCHEST/")
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--num_classes", default=14, type=int)
    parser.add_argument("--trim_data", action="store_true")

    parser.add_argument("--wandb_mode", default="offline", type=str)
    parser.add_argument("--run_name", default="NVCM", type=str)
    parser.add_argument("--exp_name", default="NVCM_NIH", type=str)
    parser.add_argument("--norm_linear", action="store_true")

    parser.add_argument("--reg_weight", default=1, type=int)
    parser.add_argument("--reg_update_beta", default=0.9, type=float)
    parser.add_argument("--nvcm", action="store_true")
    parser.add_argument("--lm", action="store_true")
    parser.add_argument("--calib_mode", default="in_mem", type=str)

    parser.add_argument("--save_dir", default="/mnt/hd/Logs/noisy_multi_label/", type=str)

    parser.add_argument("--resume", default=None, type=int)
    args = parser.parse_args()
    return args
