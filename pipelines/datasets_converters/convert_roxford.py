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


def build_roxford_df(dataset_root: Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root)

    gnd_oxford_file = dataset_root / "gnd_roxford5k.pkl"
    assert gnd_oxford_file.is_file(), f"File {gnd_oxford_file} does not exist."

    oxford_data = load_pickle_file(gnd_oxford_file)

    # We assume test data is in qimlist, train data in imlist
    oxford_imlist = oxford_data['imlist']
    oxford_qimlist = oxford_data['qimlist']

    # Extract category from image name (format: {category}_xxxxxx.jpg)
    def extract_category(name):
        return name.rsplit('_', 1)[0]

    # Get all categories
    all_names = oxford_imlist + oxford_qimlist
    all_categories = [extract_category(name) for name in all_names]
    unique_categories = sorted(set(all_categories))
    category_to_label = {cat: idx + 1 for idx, cat in enumerate(unique_categories)}

    def create_df(names, split, is_query, is_gallery):
        paths = [f"jpg/{name}.jpg" for name in names]
        categories = [extract_category(name) for name in names]
        labels = [category_to_label[cat] for cat in categories]

        return pd.DataFrame({
            LABELS_COLUMN: labels,
            PATHS_COLUMN: paths,
            SPLIT_COLUMN: split,
            IS_QUERY_COLUMN: is_query,
            IS_GALLERY_COLUMN: is_gallery,
            CATEGORIES_COLUMN: categories,
        })

    # Create dataframes
    oxford_train_df = create_df(oxford_imlist, "train", None, None)
    oxford_val_df = create_df(oxford_qimlist, "validation", True, True)

    df = pd.concat([oxford_train_df, oxford_val_df], ignore_index=True)

    # Sort by label
    df = df.sort_values(by="label", ascending=True).reset_index(drop=True)

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)

    return df


def main() -> None:
    print("Revisited Oxford dataset preparation started...")
    args = get_argparser().parse_args()
    df = build_roxford_df(args.dataset_root)
    df.to_csv(args.dataset_root / "df.csv", index=None)
    print("Revisited Oxford dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
