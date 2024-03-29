import random

import numpy as np
import pandas as pd

PAD = 0
MASK = 1


def map_column(df: pd.DataFrame, col_name: str):
    """
    Maps column values to integers
    :param df:
    :param col_name:
    :return:
    """
    # 获取指定列的唯一值并将其排序
    values = sorted(list(df[col_name].unique()))
    # 创建映射
    mapping = {k: i + 2 for i, k in enumerate(values)}
    # 逆映射
    inverse_mapping = {v: k for k, v in mapping.items()}

    # 将原始列映射为新列
    df[col_name + "_mapped"] = df[col_name].map(mapping)

    return df, mapping, inverse_mapping


def get_context(
    df: pd.DataFrame, split: str, context_size: int = 120, val_context_size: int = 5
):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows
    :param df:
    :param split:
    :param context_size:
    :param val_context_size:
    :return:
    """
    # 根据split的值确定end_index的值
    if split == "train":
        end_index = random.randint(10, df.shape[0] - val_context_size)
    elif split in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError

    # 根据end_index的值确定start_index的值，确保start_index不小于0
    start_index = max(0, end_index - context_size)
    # 取出context
    context = df[start_index:end_index]

    return context


def pad_arr(arr: np.ndarray, expected_size: int = 30):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    # 填充arr的第一维度，填充的值为arr的第一行，填充的长度为expected_size - arr.shape[0]
    # 在顶部填充
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def pad_list(list_integers, history_size: int, pad_val: int = PAD, mode="left"):
    """

    :param list_integers:
    :param history_size:
    :param pad_val:
    :param mode:
    :return:
    """
    # 填充列表
    if len(list_integers) < history_size:
        if mode == "left":
            list_integers = [pad_val] * (
                history_size - len(list_integers)
            ) + list_integers
        else:
            list_integers = list_integers + [pad_val] * (
                history_size - len(list_integers)
            )

    return list_integers


def df_to_np(df, expected_size=30):
    # 将df转为np.array
    arr = np.array(df)
    # 填充arr
    arr = pad_arr(arr, expected_size=expected_size)
    return arr


def genome_mapping(genome):
    # 将genome按照movieId和tagId排序
    genome.sort_values(by=["movieId", "tagId"], inplace=True)
    movie_genome = genome.groupby("movieId")["relevance"].agg(list).reset_index()

    movie_genome = {
        a: b for a, b in zip(movie_genome["movieId"], movie_genome["relevance"])
    }

    return movie_genome
