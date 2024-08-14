import argparse
import yaml
import subprocess

DATASET_META = {
    "HO3D": {
        "url": "data/dataset_tars/HO3D_mv_test/HO3D_mv_test-{000000..000002}.tar",
        "max_view": 5,
        "epoch_size": 2706
    },
    "DexYCB": {
        "url": "data/dataset_tars/DexYCB_mv/DexYCB_mv_test-{000000..000003}.tar",
        "max_view": 8,
        "epoch_size": 4950
    },
    "Arctic": {
        "url": "data/dataset_tars/Arctic_mv/Arctic_mv_val_p1-{000000..000045}.tar",
        "max_view": 8,
        "epoch_size": 17392
    },
    "Interhand": {
        "url": "data/dataset_tars/Interhand_mv/Interhand_mv_val-{000000..000022}.tar",
        "max_view": 8,
        "epoch_size": 85255
    },
    "Oakink": {
        "url": "data/dataset_tars/Oakink_mv/Oakink_mv_test-{000000..000045}.tar",
        "max_view": 4,
        "epoch_size": 21351
    },
    "Freihand": {
        "url": "data/dataset_tars/Freihand_mv/Freihand_mv_test-{000000..000000}.tar",
        "max_view": 1,
        "epoch_size": 3960
    }
}

MODEL_CATEGORY = ['small', 'medium', 'large', 'huge', 'medium_MANO']
EMBED_SIZE = [128, 256, 512, 1024, 256]


def main(args):
    cfg_path = args.cfg
    dataset = args.dataset
    model_type = args.model
    gpu_id = args.gpu_id
    view_range = [args.view_min, args.view_max]
    reload = args.reload
    port = args.port

    if dataset not in DATASET_META:
        print(f"Dataset {dataset} not found in dataset_info.")
        assert False
    if model_type not in MODEL_CATEGORY:
        print(f"Model category {model_type} not found in model_category.")
        assert False

    # Read the cfg and change some attributes, then save at the original path
    # Parse the yaml file
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg["DATASET"]["TEST"]["TARGET"]["URLS"] = DATASET_META[dataset]["url"]
    cfg["DATASET"]["TEST"]["EPOCH_SIZE"] = DATASET_META[dataset]["epoch_size"]
    cfg["DATASET"]["TEST"]["TARGET"]["EPOCH_SIZE"] = DATASET_META[dataset]["epoch_size"]

    # Set view range for Freihand
    if dataset == "Freihand":
        view_range = [1, 1]
        print("Setting view range to 1 for Freihand dataset.")

    cfg["DATASET"]["TEST"]["TARGET"]["VIEW_RANGE"] = view_range

    # set the embed size based on the model type
    model_idx = MODEL_CATEGORY.index(model_type)
    embed_size = EMBED_SIZE[model_idx]
    cfg["MODEL"]["HEAD"]["POSITIONAL_ENCODING"]["NUM_FEATS"] = embed_size // 2
    cfg["MODEL"]["HEAD"]["TRANSFORMER"]["INPUT_FEAT_DIM"] = embed_size
    cfg["MODEL"]["HEAD"]["POINTS_FEAT_DIM"] = embed_size
    cfg["MODEL"]["HEAD"]["EMBED_DIMS"] = embed_size

    # Set the parametric output for medium_MANO
    if model_type == "medium_MANO":
        cfg["MODEL"]["HEAD"]["TRANSFORMER"]["PARAMETRIC_OUTPUT"] = True
    else:
        cfg["MODEL"]["HEAD"]["TRANSFORMER"]["PARAMETRIC_OUTPUT"] = False

    # Dump the yaml file in the original path
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    # Run the eval.py using shell
    eval_extra = "draw" if args.draw else "auc"
    command = f"./ddp_python scripts/eval.py --cfg {cfg_path} -g {gpu_id} -w 1 --exp_id {dataset}_view_{view_range[0]}_{view_range[1]}_{model_type} --reload {reload} -p {port} --eval_extra {eval_extra}"
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Error in running {command}")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Single Setting")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--view_min", type=int, required=True, help="Minimum view range.")
    parser.add_argument("--view_max", type=int, required=True, help="Maximum view range.")
    parser.add_argument("--model", type=str, required=True, help="Model category.")
    parser.add_argument("--gpu_id", "-g", type=int, default=0, required=True, help="GPU ID to run the evaluation.")
    parser.add_argument("--reload", type=str, default=None, help="Path to the checkpoint to reload.")
    parser.add_argument("--port", "-p", type=int, default=60000, help="Port to run the evaluation.")
    parser.add_argument("--draw", "-d", action="store_true", help="Visualize the results.")
    args = parser.parse_args()
    main(args)
