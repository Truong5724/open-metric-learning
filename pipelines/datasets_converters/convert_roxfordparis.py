from argparse import ArgumentParser
from pathlib import Path

import pickle
import pandas as pd

from oml.const import (
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    PATHS_COLUMN,
    SPLIT_COLUMN,
)

from oml.utils.dataframe_format import check_retrieval_dataframe_format


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    return parser


def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def build_roxfordparis_df(dataset_root: Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root)

    gnd_oxford_file = dataset_root / "roxford5k" / "gnd_roxford5k.pkl"
    gnd_paris_file = dataset_root / "rparis6k" / "gnd_rparis6k.pkl"

    for file in [gnd_oxford_file, gnd_paris_file]:
        assert file.is_file(), f"File {file} does not exist."

    oxford_data = load_pickle_file(gnd_oxford_file)
    paris_data = load_pickle_file(gnd_paris_file)

    # We assume test data is in qimlist, train data in imlist
    oxford_imlist = oxford_data['imlist']
    paris_imlist = paris_data['imlist']
    oxford_qimlist = oxford_data['qimlist']
    paris_qimlist = paris_data['qimlist']

    # Extract category from image name (format: {category}_xxxxxx.jpg)
    def extract_category(name):
        return name.rsplit('_', 1)[0]

    # Get all categories
    all_names = oxford_imlist + paris_imlist + oxford_qimlist + paris_qimlist
    all_categories = [extract_category(name) for name in all_names]
    unique_categories = sorted(set(all_categories))
    category_to_label = {cat: idx for idx, cat in enumerate(unique_categories)}

    def create_df(names, prefix, split, is_query, is_gallery):
        paths = [f"{prefix}/jpg/{name}.jpg" for name in names]
        categories = [extract_category(name) for name in names]
        labels = [category_to_label[cat] for cat in categories]

        return pd.DataFrame({
            PATHS_COLUMN: paths,
            LABELS_COLUMN: labels,
            SPLIT_COLUMN: split,
            IS_QUERY_COLUMN: is_query,
            IS_GALLERY_COLUMN: is_gallery,
            CATEGORIES_COLUMN: categories,
        })

    # Create dataframes
    oxford_train_df = create_df(oxford_imlist, "roxford5k", "train", None, None)
    paris_train_df = create_df(paris_imlist, "rparis6k", "train", None, None)
    oxford_val_df = create_df(oxford_qimlist, "roxford5k", "validation", True, True)
    paris_val_df = create_df(paris_qimlist, "rparis6k", "validation", True, True)

    df = pd.concat([oxford_train_df, paris_train_df, oxford_val_df, paris_val_df], ignore_index=True)

    # Sort by label
    df = df.sort_values(by="label", ascending=True).reset_index(drop=True)

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)

    return df


def main() -> None:
    print("Oxford5k dataset preparation started...")
    args = get_argparser().parse_args()
    df = build_roxfordparis_df(args.dataset_root)
    df.to_csv(args.dataset_root / df.csv, index=None)
    print("Oxford5k dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
