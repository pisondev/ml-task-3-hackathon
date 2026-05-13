"""Repeatable KNN and Naive Bayes experiments for Spaceship Titanic.

The course rule allows final predictions only from KNeighborsClassifier and
Naive Bayes variants. This script uses richer preprocessing, but every final
estimator evaluated and submitted is one of those allowed model classes.
"""

from __future__ import annotations

import argparse
import inspect
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings(
    "ignore",
    message="Bins whose width are too small.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The current default behavior, quantile_method='linear'.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


RANDOM_STATE = 42
N_SPLITS = 5

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data-from-kaggle"
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
EXPERIMENTS_PATH = REPORTS_DIR / "experiments.csv"
BEST_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_best.csv"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

CATEGORICAL_COLS = [
    "HomePlanet",
    "Destination",
    "CryoSleep",
    "VIP",
    "CabinDeck",
    "CabinSide",
    "CabinRegion",
    "DeckSide",
    "AgeBand",
    "HomeDestination",
]

SKEWED_NUMERIC_COLS = SPEND_COLS + [
    "TotalSpend",
    "LuxurySpend",
    "ServiceSpend",
    "SpendPerPerson",
    "SpendPerAge",
]

NUMERIC_COLS = [
    "Age",
    "GroupNumber",
    "PassengerNumber",
    "GroupSize",
    "CabinNumber",
    "SurnameSize",
    "NameLength",
    "SpendMissingCount",
    "CabinNumberMissing",
    "SpendShareLuxury",
    "SpendShareService",
]

BINARY_COLS = [
    "IsSolo",
    "HasName",
    "HasCabin",
    "AgeMissing",
    "IsChild",
    "IsTeen",
    "IsAdult",
    "IsSenior",
    "CryoSleepKnown",
    "CryoSleepFlag",
    "VIPKnown",
    "VIPFlag",
    "NoSpend",
    "AnySpend",
    "HasRoomService",
    "HasFoodCourt",
    "HasShoppingMall",
    "HasSpa",
    "HasVRDeck",
    "HasLuxurySpend",
    "HasServiceSpend",
    "CryoWithSpend",
    "CryoSleepInferred",
    "VIPInferred",
    "CryoUnknownNoSpend",
    "VIPUnknown",
]

BERNOULLI_FEATURE_COLS = CATEGORICAL_COLS + BINARY_COLS
CATEGORICAL_NB_COLS = CATEGORICAL_COLS + BINARY_COLS
CATEGORICAL_NB_NUMERIC_COLS = SKEWED_NUMERIC_COLS + NUMERIC_COLS


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder across old and new sklearn versions."""

    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def _clip_log1p(values: np.ndarray) -> np.ndarray:
    """Use log1p for skewed non-negative spend features after imputation."""

    return np.log1p(np.clip(values.astype(float), 0.0, None))


def _to_int(values: np.ndarray) -> np.ndarray:
    return values.astype(np.int64)


def _shift_ordinal(values: np.ndarray) -> np.ndarray:
    """Reserve 0 for unknown categories, shift known categories to 1..n."""

    return values.astype(np.int64) + 1


def _bool_to_float(series: pd.Series) -> pd.Series:
    mapped = series.astype("string").str.lower().map({"true": 1.0, "false": 0.0})
    return mapped.astype(float)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


class SpaceshipFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add target-free features from PassengerId, Cabin, Name, spend, and age."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SpaceshipFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        passenger = df["PassengerId"].astype("string").str.split("_", expand=True)
        df["GroupId"] = passenger[0].fillna("0")
        df["GroupNumber"] = _safe_numeric(df["GroupId"])
        df["PassengerNumber"] = _safe_numeric(passenger[1] if passenger.shape[1] > 1 else pd.Series(index=df.index))
        group_sizes = df["GroupId"].map(df["GroupId"].value_counts())
        df["GroupSize"] = group_sizes.astype(float)
        df["IsSolo"] = (df["GroupSize"] <= 1).astype(float)

        cabin = df["Cabin"].astype("string").str.split("/", expand=True)
        df["CabinDeck"] = cabin[0].fillna("Unknown")
        df["CabinNumber"] = _safe_numeric(cabin[1] if cabin.shape[1] > 1 else pd.Series(index=df.index))
        df["CabinSide"] = (cabin[2] if cabin.shape[1] > 2 else pd.Series(index=df.index)).fillna("Unknown")
        df["HasCabin"] = df["Cabin"].notna().astype(float)
        df["CabinNumberMissing"] = df["CabinNumber"].isna().astype(float)
        cabin_number_for_region = df["CabinNumber"].fillna(-1)
        df["CabinRegion"] = pd.cut(
            cabin_number_for_region,
            bins=[-2, -0.5, 299, 599, 899, 1199, 1499, np.inf],
            labels=["Unknown", "000-299", "300-599", "600-899", "900-1199", "1200-1499", "1500+"],
        ).astype("string")
        df["DeckSide"] = df["CabinDeck"].fillna("Unknown") + "_" + df["CabinSide"].fillna("Unknown")

        name = df["Name"].astype("string")
        df["HasName"] = name.notna().astype(float)
        df["NameLength"] = name.fillna("").str.len().astype(float)
        surname = name.str.split().str[-1].fillna("Unknown")
        surname_sizes = surname.map(surname.value_counts())
        df["SurnameSize"] = surname_sizes.astype(float)

        for col in SPEND_COLS:
            df[col] = _safe_numeric(df[col])
            df[f"Has{col}"] = (df[col].fillna(0) > 0).astype(float)
        spend_values = df[SPEND_COLS]
        df["SpendMissingCount"] = spend_values.isna().sum(axis=1).astype(float)
        spend_filled = spend_values.fillna(0)
        df["TotalSpend"] = spend_filled.sum(axis=1)
        df["LuxurySpend"] = spend_filled[["FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
        df["ServiceSpend"] = spend_filled[["RoomService", "Spa", "VRDeck"]].sum(axis=1)
        df["SpendPerPerson"] = df["TotalSpend"] / df["GroupSize"].replace(0, np.nan)
        df["NoSpend"] = (df["TotalSpend"] <= 0).astype(float)
        df["AnySpend"] = (df["TotalSpend"] > 0).astype(float)
        df["HasLuxurySpend"] = (df["LuxurySpend"] > 0).astype(float)
        df["HasServiceSpend"] = (df["ServiceSpend"] > 0).astype(float)
        total_spend_denominator = df["TotalSpend"].replace(0, np.nan)
        df["SpendShareLuxury"] = (df["LuxurySpend"] / total_spend_denominator).fillna(0)
        df["SpendShareService"] = (df["ServiceSpend"] / total_spend_denominator).fillna(0)

        df["CryoSleepKnown"] = df["CryoSleep"].notna().astype(float)
        df["CryoSleepFlag"] = _bool_to_float(df["CryoSleep"])
        df["VIPKnown"] = df["VIP"].notna().astype(float)
        df["VIPFlag"] = _bool_to_float(df["VIP"])
        df["CryoWithSpend"] = ((df["CryoSleepFlag"] == 1.0) & (df["TotalSpend"] > 0)).astype(float)
        df["CryoSleepInferred"] = df["CryoSleepFlag"].fillna(df["NoSpend"]).astype(float)
        df["VIPInferred"] = df["VIPFlag"].fillna(0).astype(float)
        df["CryoUnknownNoSpend"] = ((df["CryoSleep"].isna()) & (df["NoSpend"] == 1)).astype(float)
        df["VIPUnknown"] = df["VIP"].isna().astype(float)

        df["Age"] = _safe_numeric(df["Age"])
        df["AgeMissing"] = df["Age"].isna().astype(float)
        df["SpendPerAge"] = df["TotalSpend"] / (df["Age"].fillna(df["Age"].median()) + 1)
        age = df["Age"]
        df["IsChild"] = (age < 13).fillna(False).astype(float)
        df["IsTeen"] = ((age >= 13) & (age < 18)).fillna(False).astype(float)
        df["IsAdult"] = ((age >= 18) & (age < 60)).fillna(False).astype(float)
        df["IsSenior"] = (age >= 60).fillna(False).astype(float)
        df["AgeBand"] = pd.cut(
            age,
            bins=[-np.inf, 12, 17, 29, 44, 59, np.inf],
            labels=["child", "teen", "young_adult", "adult", "older_adult", "senior"],
        ).astype("string").fillna("Unknown")

        df["HomeDestination"] = (
            df["HomePlanet"].astype("string").fillna("Unknown")
            + "_"
            + df["Destination"].astype("string").fillna("Unknown")
        )

        return df


@dataclass(frozen=True)
class Experiment:
    name: str
    family: str
    pipeline: Pipeline
    batch: str


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    y = train["Transported"].astype(bool)
    X = train.drop(columns=["Transported"])
    return X, y, test, sample_submission


def make_knn_preprocessor(
    scaler: BaseEstimator,
    *,
    skewed_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    binary_cols: list[str] | None = None,
    transformer_weights: dict[str, float] | None = None,
) -> ColumnTransformer:
    skewed_cols = skewed_cols or SKEWED_NUMERIC_COLS
    numeric_cols = numeric_cols or NUMERIC_COLS
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    binary_cols = binary_cols or BINARY_COLS

    scaled_skewed = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(_clip_log1p, validate=False)),
            ("scaler", scaler),
        ]
    )
    scaled_numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    binary = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        [
            ("skewed", scaled_skewed, skewed_cols),
            ("numeric", scaled_numeric, numeric_cols),
            ("categorical", categorical, categorical_cols),
            ("binary", binary, binary_cols),
        ],
        remainder="drop",
        transformer_weights=transformer_weights,
    )


def make_knn_quantile_preprocessor(
    *,
    skewed_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    binary_cols: list[str] | None = None,
    transformer_weights: dict[str, float] | None = None,
    output_distribution: str = "normal",
) -> ColumnTransformer:
    skewed_cols = skewed_cols or SKEWED_NUMERIC_COLS
    numeric_cols = numeric_cols or NUMERIC_COLS
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    binary_cols = binary_cols or BINARY_COLS

    quantile = QuantileTransformer(
        n_quantiles=256,
        output_distribution=output_distribution,
        random_state=RANDOM_STATE,
        subsample=None,
    )
    skewed = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(_clip_log1p, validate=False)),
            ("quantile", quantile),
        ]
    )
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=256,
                    output_distribution=output_distribution,
                    random_state=RANDOM_STATE,
                    subsample=None,
                ),
            ),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    binary = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    return ColumnTransformer(
        [
            ("skewed", skewed, skewed_cols),
            ("numeric", numeric, numeric_cols),
            ("categorical", categorical, categorical_cols),
            ("binary", binary, binary_cols),
        ],
        remainder="drop",
        transformer_weights=transformer_weights,
    )


def make_gaussian_nb_preprocessor(
    *,
    skewed_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    binary_cols: list[str] | None = None,
) -> ColumnTransformer:
    skewed_cols = skewed_cols or SKEWED_NUMERIC_COLS
    numeric_cols = numeric_cols or NUMERIC_COLS
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    binary_cols = binary_cols or BINARY_COLS

    skewed = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(_clip_log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    binary = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        [
            ("skewed", skewed, skewed_cols),
            ("numeric", numeric, numeric_cols),
            ("categorical", categorical, categorical_cols),
            ("binary", binary, binary_cols),
        ],
        remainder="drop",
    )


def make_bernoulli_nb_preprocessor(
    *,
    categorical_cols: list[str] | None = None,
    binary_cols: list[str] | None = None,
) -> ColumnTransformer:
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    binary_cols = binary_cols or BINARY_COLS

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    binary = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        [
            ("categorical", categorical, categorical_cols),
            ("binary", binary, binary_cols),
        ],
        remainder="drop",
    )


def make_categorical_nb_preprocessor(
    *,
    categorical_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    n_bins: int = 8,
    strategy: str = "quantile",
) -> ColumnTransformer:
    categorical_cols = categorical_cols or CATEGORICAL_NB_COLS
    numeric_cols = numeric_cols or CATEGORICAL_NB_NUMERIC_COLS

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
            ("shift", FunctionTransformer(_shift_ordinal, validate=False)),
        ]
    )
    numeric_bins = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "bins",
                KBinsDiscretizer(
                    n_bins=n_bins,
                    encode="ordinal",
                    strategy=strategy,
                    subsample=None,
                ),
            ),
            ("int", FunctionTransformer(_to_int, validate=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("categorical", categorical, categorical_cols),
            ("numeric_bins", numeric_bins, numeric_cols),
        ],
        remainder="drop",
    )


def make_binned_onehot_nb_preprocessor(
    *,
    categorical_cols: list[str],
    binary_cols: list[str],
    numeric_cols: list[str],
    n_bins: int,
) -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    binary = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    numeric_bins = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "bins",
                KBinsDiscretizer(
                    n_bins=n_bins,
                    encode="onehot-dense",
                    strategy="quantile",
                    subsample=None,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("categorical", categorical, categorical_cols),
            ("binary", binary, binary_cols),
            ("numeric_bins", numeric_bins, numeric_cols),
        ],
        remainder="drop",
    )


def make_pipeline(preprocessor: ColumnTransformer, estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("features", SpaceshipFeatureEngineer()),
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def build_baseline_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []

    scaler_factories = {
        "standard": StandardScaler,
        "robust": RobustScaler,
    }
    for scaler_name, scaler_factory in scaler_factories.items():
        for n_neighbors in [15, 25, 35, 50, 75]:
            for weights in ["uniform", "distance"]:
                for p in [1, 2]:
                    model = KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        p=p,
                    )
                    name = (
                        f"knn_{scaler_name}_k{n_neighbors}_"
                        f"{weights}_p{p}"
                    )
                    experiments.append(
                        Experiment(
                            name=name,
                            family="KNN",
                            batch="baseline_all",
                            pipeline=make_pipeline(
                                make_knn_preprocessor(scaler_factory()),
                                model,
                            ),
                        )
                    )

    for var_smoothing in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        experiments.append(
            Experiment(
                name=f"gaussian_nb_vs{var_smoothing:g}",
                family="GaussianNB",
                batch="baseline_all",
                pipeline=make_pipeline(
                    make_gaussian_nb_preprocessor(),
                    GaussianNB(var_smoothing=var_smoothing),
                ),
            )
        )

    for alpha in [0.25, 0.5, 1.0, 2.0]:
        experiments.append(
            Experiment(
                name=f"bernoulli_nb_alpha{alpha:g}",
                family="BernoulliNB",
                batch="baseline_all",
                pipeline=make_pipeline(
                    make_bernoulli_nb_preprocessor(),
                    BernoulliNB(alpha=alpha),
                ),
            )
        )

    for alpha in [0.25, 0.5, 1.0, 2.0]:
        experiments.append(
            Experiment(
                name=f"categorical_nb_alpha{alpha:g}",
                family="CategoricalNB",
                batch="baseline_all",
                pipeline=make_pipeline(
                    make_categorical_nb_preprocessor(),
                    CategoricalNB(alpha=alpha),
                ),
            )
        )

    return experiments


def build_knn_k_refine_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    for n_neighbors in [19, 21, 23, 25, 27, 29, 31, 33]:
        for weights in ["uniform", "distance"]:
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=2)
            name = f"knn_k_refine_standard_k{n_neighbors}_{weights}_p2"
            experiments.append(
                Experiment(
                    name=name,
                    family="KNN",
                    batch="knn_k_refine",
                    pipeline=make_pipeline(make_knn_preprocessor(StandardScaler()), model),
                )
            )
    return experiments


def build_knn_scaler_compare_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    scaler_factories = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }
    for scaler_name, scaler_factory in scaler_factories.items():
        for n_neighbors in [21, 25, 29]:
            for p in [1, 2]:
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights="uniform",
                    p=p,
                )
                name = f"knn_scaler_{scaler_name}_k{n_neighbors}_uniform_p{p}"
                experiments.append(
                    Experiment(
                        name=name,
                        family="KNN",
                        batch="knn_scaler_compare",
                        pipeline=make_pipeline(
                            make_knn_preprocessor(scaler_factory()),
                            model,
                        ),
                    )
                )
    return experiments


def build_knn_feature_trim_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    numeric_no_ids = [
        col
        for col in NUMERIC_COLS
        if col not in {"GroupNumber", "CabinNumber", "NameLength"}
    ]
    numeric_no_name = [
        col
        for col in NUMERIC_COLS
        if col not in {"SurnameSize", "NameLength"}
    ]
    skewed_core = SPEND_COLS + ["TotalSpend", "SpendPerPerson", "SpendPerAge"]
    variants = {
        "no_ids": {
            "numeric_cols": numeric_no_ids,
            "skewed_cols": SKEWED_NUMERIC_COLS,
            "categorical_cols": CATEGORICAL_COLS,
            "binary_cols": BINARY_COLS,
        },
        "no_name": {
            "numeric_cols": numeric_no_name,
            "skewed_cols": SKEWED_NUMERIC_COLS,
            "categorical_cols": CATEGORICAL_COLS,
            "binary_cols": BINARY_COLS,
        },
        "core_spend": {
            "numeric_cols": numeric_no_ids,
            "skewed_cols": skewed_core,
            "categorical_cols": CATEGORICAL_COLS,
            "binary_cols": BINARY_COLS,
        },
    }
    for variant_name, cols in variants.items():
        for n_neighbors in [23, 25, 27, 29]:
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights="uniform",
                p=2,
            )
            name = f"knn_feature_{variant_name}_standard_k{n_neighbors}_uniform_p2"
            experiments.append(
                Experiment(
                    name=name,
                    family="KNN",
                    batch="knn_feature_trim",
                    pipeline=make_pipeline(
                        make_knn_preprocessor(StandardScaler(), **cols),
                        model,
                    ),
                )
            )
    return experiments


def build_knn_no_name_refine_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    numeric_no_name = [
        col
        for col in NUMERIC_COLS
        if col not in {"SurnameSize", "NameLength"}
    ]
    for n_neighbors in [19, 21, 23, 25, 27, 29, 31, 33]:
        for weights in ["uniform", "distance"]:
            for p in [1, 2]:
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    p=p,
                )
                name = f"knn_no_name_refine_k{n_neighbors}_{weights}_p{p}"
                experiments.append(
                    Experiment(
                        name=name,
                        family="KNN",
                        batch="knn_no_name_refine",
                        pipeline=make_pipeline(
                            make_knn_preprocessor(
                                StandardScaler(),
                                numeric_cols=numeric_no_name,
                            ),
                            model,
                        ),
                    )
                )
    return experiments


def build_knn_no_name_fine_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    numeric_no_name = [
        col
        for col in NUMERIC_COLS
        if col not in {"SurnameSize", "NameLength"}
    ]
    for n_neighbors in range(20, 27):
        for weights in ["uniform", "distance"]:
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                p=1,
            )
            name = f"knn_no_name_fine_k{n_neighbors}_{weights}_p1"
            experiments.append(
                Experiment(
                    name=name,
                    family="KNN",
                    batch="knn_no_name_fine",
                    pipeline=make_pipeline(
                        make_knn_preprocessor(
                            StandardScaler(),
                            numeric_cols=numeric_no_name,
                        ),
                        model,
                    ),
                )
            )
    return experiments


def build_knn_feature_weight_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    numeric_no_name = [
        col
        for col in NUMERIC_COLS
        if col not in {"SurnameSize", "NameLength"}
    ]
    weight_variants = {
        "cat075_bin100": {"categorical": 0.75, "binary": 1.0},
        "cat050_bin100": {"categorical": 0.50, "binary": 1.0},
        "cat125_bin100": {"categorical": 1.25, "binary": 1.0},
        "cat100_bin075": {"categorical": 1.0, "binary": 0.75},
        "cat100_bin125": {"categorical": 1.0, "binary": 1.25},
        "cat075_bin125": {"categorical": 0.75, "binary": 1.25},
        "cat125_bin075": {"categorical": 1.25, "binary": 0.75},
        "skew125_num100": {"skewed": 1.25, "numeric": 1.0},
        "skew075_num100": {"skewed": 0.75, "numeric": 1.0},
        "skew100_num075": {"skewed": 1.0, "numeric": 0.75},
        "skew100_num125": {"skewed": 1.0, "numeric": 1.25},
    }
    for variant_name, weights in weight_variants.items():
        transformer_weights = {
            "skewed": weights.get("skewed", 1.0),
            "numeric": weights.get("numeric", 1.0),
            "categorical": weights.get("categorical", 1.0),
            "binary": weights.get("binary", 1.0),
        }
        for n_neighbors in [21, 23, 25]:
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights="distance",
                p=1,
            )
            name = f"knn_weight_{variant_name}_k{n_neighbors}_distance_p1"
            experiments.append(
                Experiment(
                    name=name,
                    family="KNN",
                    batch="knn_feature_weights",
                    pipeline=make_pipeline(
                        make_knn_preprocessor(
                            StandardScaler(),
                            numeric_cols=numeric_no_name,
                            transformer_weights=transformer_weights,
                        ),
                        model,
                    ),
                )
            )
    return experiments


def build_knn_weight_fine_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    numeric_no_name = [
        col
        for col in NUMERIC_COLS
        if col not in {"SurnameSize", "NameLength"}
    ]
    weight_variants = {
        "cat100_bin065_num100_skew100": {
            "categorical": 1.00,
            "binary": 0.65,
            "numeric": 1.00,
            "skewed": 1.00,
        },
        "cat100_bin075_num115_skew100": {
            "categorical": 1.00,
            "binary": 0.75,
            "numeric": 1.15,
            "skewed": 1.00,
        },
        "cat100_bin075_num125_skew100": {
            "categorical": 1.00,
            "binary": 0.75,
            "numeric": 1.25,
            "skewed": 1.00,
        },
        "cat100_bin075_num135_skew100": {
            "categorical": 1.00,
            "binary": 0.75,
            "numeric": 1.35,
            "skewed": 1.00,
        },
        "cat090_bin075_num125_skew100": {
            "categorical": 0.90,
            "binary": 0.75,
            "numeric": 1.25,
            "skewed": 1.00,
        },
        "cat110_bin075_num125_skew100": {
            "categorical": 1.10,
            "binary": 0.75,
            "numeric": 1.25,
            "skewed": 1.00,
        },
        "cat100_bin075_num125_skew110": {
            "categorical": 1.00,
            "binary": 0.75,
            "numeric": 1.25,
            "skewed": 1.10,
        },
        "cat100_bin075_num125_skew090": {
            "categorical": 1.00,
            "binary": 0.75,
            "numeric": 1.25,
            "skewed": 0.90,
        },
    }
    for variant_name, transformer_weights in weight_variants.items():
        for n_neighbors in [21, 23, 25]:
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights="distance",
                p=1,
            )
            name = f"knn_weight_fine_{variant_name}_k{n_neighbors}_distance_p1"
            experiments.append(
                Experiment(
                    name=name,
                    family="KNN",
                    batch="knn_weight_fine",
                    pipeline=make_pipeline(
                        make_knn_preprocessor(
                            StandardScaler(),
                            numeric_cols=numeric_no_name,
                            transformer_weights=transformer_weights,
                        ),
                        model,
                    ),
                )
            )
    return experiments


def build_knn_weight_micro_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    numeric_no_name = [
        col
        for col in NUMERIC_COLS
        if col not in {"SurnameSize", "NameLength"}
    ]
    for binary_weight in [0.70, 0.75, 0.80]:
        for numeric_weight in [1.20, 1.25, 1.30]:
            transformer_weights = {
                "skewed": 1.00,
                "numeric": numeric_weight,
                "categorical": 1.00,
                "binary": binary_weight,
            }
            for n_neighbors in [22, 23, 24]:
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights="distance",
                    p=1,
                )
                variant = (
                    f"bin{int(binary_weight * 100):03d}_"
                    f"num{int(numeric_weight * 100):03d}"
                )
                name = f"knn_weight_micro_{variant}_k{n_neighbors}_distance_p1"
                experiments.append(
                    Experiment(
                        name=name,
                        family="KNN",
                        batch="knn_weight_micro",
                        pipeline=make_pipeline(
                            make_knn_preprocessor(
                                StandardScaler(),
                                numeric_cols=numeric_no_name,
                                transformer_weights=transformer_weights,
                            ),
                            model,
                        ),
                    )
                )
    return experiments


def build_knn_weight_ablation_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    base_numeric = [
        col
        for col in NUMERIC_COLS
        if col not in {"SurnameSize", "NameLength"}
    ]
    base_skewed = SKEWED_NUMERIC_COLS
    base_categorical = CATEGORICAL_COLS
    base_binary = BINARY_COLS
    variants = {
        "drop_group_number": {
            "numeric_cols": [col for col in base_numeric if col != "GroupNumber"],
            "skewed_cols": base_skewed,
            "categorical_cols": base_categorical,
            "binary_cols": base_binary,
        },
        "drop_cabin_number": {
            "numeric_cols": [col for col in base_numeric if col != "CabinNumber"],
            "skewed_cols": base_skewed,
            "categorical_cols": base_categorical,
            "binary_cols": base_binary,
        },
        "drop_passenger_number": {
            "numeric_cols": [col for col in base_numeric if col != "PassengerNumber"],
            "skewed_cols": base_skewed,
            "categorical_cols": base_categorical,
            "binary_cols": base_binary,
        },
        "drop_ratio_numeric": {
            "numeric_cols": [
                col
                for col in base_numeric
                if col not in {"SpendShareLuxury", "SpendShareService"}
            ],
            "skewed_cols": [
                col
                for col in base_skewed
                if col not in {"SpendPerPerson", "SpendPerAge"}
            ],
            "categorical_cols": base_categorical,
            "binary_cols": base_binary,
        },
        "drop_interaction_cats": {
            "numeric_cols": base_numeric,
            "skewed_cols": base_skewed,
            "categorical_cols": [
                col
                for col in base_categorical
                if col not in {"DeckSide", "HomeDestination"}
            ],
            "binary_cols": base_binary,
        },
        "drop_inferred_flags": {
            "numeric_cols": base_numeric,
            "skewed_cols": base_skewed,
            "categorical_cols": base_categorical,
            "binary_cols": [
                col
                for col in base_binary
                if col
                not in {
                    "CryoSleepInferred",
                    "VIPInferred",
                    "CryoUnknownNoSpend",
                    "VIPUnknown",
                    "CryoWithSpend",
                }
            ],
        },
        "drop_spend_flags": {
            "numeric_cols": base_numeric,
            "skewed_cols": base_skewed,
            "categorical_cols": base_categorical,
            "binary_cols": [
                col
                for col in base_binary
                if col
                not in {
                    "AnySpend",
                    "NoSpend",
                    "HasRoomService",
                    "HasFoodCourt",
                    "HasShoppingMall",
                    "HasSpa",
                    "HasVRDeck",
                    "HasLuxurySpend",
                    "HasServiceSpend",
                }
            ],
        },
    }
    weight_variants = {
        "bin070_num120": {"skewed": 1.0, "numeric": 1.20, "categorical": 1.0, "binary": 0.70},
        "bin070_num125": {"skewed": 1.0, "numeric": 1.25, "categorical": 1.0, "binary": 0.70},
    }
    for weight_name, transformer_weights in weight_variants.items():
        for variant_name, cols in variants.items():
            model = KNeighborsClassifier(n_neighbors=23, weights="distance", p=1)
            name = f"knn_ablate_{variant_name}_{weight_name}_k23_distance_p1"
            experiments.append(
                Experiment(
                    name=name,
                    family="KNN",
                    batch="knn_weight_ablation",
                    pipeline=make_pipeline(
                        make_knn_preprocessor(
                            StandardScaler(),
                            transformer_weights=transformer_weights,
                            **cols,
                        ),
                        model,
                    ),
                )
            )
    return experiments


def build_knn_quantile_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    numeric_no_name = [
        col
        for col in NUMERIC_COLS
        if col not in {"SurnameSize", "NameLength"}
    ]
    weight_variants = {
        "bin070_num120": {"skewed": 1.0, "numeric": 1.20, "categorical": 1.0, "binary": 0.70},
        "bin070_num125": {"skewed": 1.0, "numeric": 1.25, "categorical": 1.0, "binary": 0.70},
        "bin080_num120": {"skewed": 1.0, "numeric": 1.20, "categorical": 1.0, "binary": 0.80},
    }
    for output_distribution in ["normal", "uniform"]:
        for weight_name, transformer_weights in weight_variants.items():
            for n_neighbors in [21, 23, 25]:
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights="distance",
                    p=1,
                )
                name = (
                    f"knn_quantile_{output_distribution}_{weight_name}_"
                    f"k{n_neighbors}_distance_p1"
                )
                experiments.append(
                    Experiment(
                        name=name,
                        family="KNN",
                        batch="knn_quantile",
                        pipeline=make_pipeline(
                            make_knn_quantile_preprocessor(
                                numeric_cols=numeric_no_name,
                                transformer_weights=transformer_weights,
                                output_distribution=output_distribution,
                            ),
                            model,
                        ),
                    )
                )
    return experiments


def build_nb_lowcard_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    lowcard_cats = [
        "HomePlanet",
        "Destination",
        "CryoSleep",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "CabinRegion",
        "AgeBand",
    ]
    lowcard_with_deck_side = lowcard_cats + ["DeckSide"]
    compact_numeric = [
        "Age",
        "GroupSize",
        "PassengerNumber",
        "CabinNumber",
        "TotalSpend",
        "SpendPerPerson",
        "SpendPerAge",
        "SpendShareLuxury",
        "SpendShareService",
    ]
    categorical_variants = {
        "lowcard": lowcard_cats + BINARY_COLS,
        "deck_side": lowcard_with_deck_side + BINARY_COLS,
    }
    for variant_name, cats in categorical_variants.items():
        for n_bins in [4, 6, 8]:
            for alpha in [0.5, 1.0, 2.0]:
                name = f"categorical_nb_{variant_name}_bins{n_bins}_alpha{alpha:g}"
                experiments.append(
                    Experiment(
                        name=name,
                        family="CategoricalNB",
                        batch="nb_lowcard",
                        pipeline=make_pipeline(
                            make_categorical_nb_preprocessor(
                                categorical_cols=cats,
                                numeric_cols=compact_numeric,
                                n_bins=n_bins,
                            ),
                            CategoricalNB(alpha=alpha),
                        ),
                    )
                )
    for alpha in [0.25, 0.5, 1.0, 2.0]:
        name = f"bernoulli_nb_lowcard_alpha{alpha:g}"
        experiments.append(
            Experiment(
                name=name,
                family="BernoulliNB",
                batch="nb_lowcard",
                pipeline=make_pipeline(
                    make_bernoulli_nb_preprocessor(categorical_cols=lowcard_with_deck_side),
                    BernoulliNB(alpha=alpha),
                ),
            )
        )
    return experiments


def build_nb_binned_onehot_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    lowcard = [
        "HomePlanet",
        "Destination",
        "CryoSleep",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "CabinRegion",
        "AgeBand",
    ]
    categorical_sets = {
        "lowcard": lowcard,
        "full_lowish": CATEGORICAL_COLS,
    }
    numeric_cols = [
        "Age",
        "GroupSize",
        "PassengerNumber",
        "CabinNumber",
        "TotalSpend",
        "LuxurySpend",
        "ServiceSpend",
        "SpendPerPerson",
        "SpendPerAge",
        "SpendShareLuxury",
        "SpendShareService",
    ]
    model_factories = {
        "bernoulli": BernoulliNB,
        "multinomial": MultinomialNB,
        "complement": ComplementNB,
    }
    for cat_name, categorical_cols in categorical_sets.items():
        for n_bins in [4, 8]:
            for model_name, model_factory in model_factories.items():
                for alpha in [0.5, 1.0]:
                    name = f"{model_name}_nb_binned_{cat_name}_bins{n_bins}_alpha{alpha:g}"
                    experiments.append(
                        Experiment(
                            name=name,
                            family=f"{model_factory.__name__}",
                            batch="nb_binned_onehot",
                            pipeline=make_pipeline(
                                make_binned_onehot_nb_preprocessor(
                                    categorical_cols=categorical_cols,
                                    binary_cols=BINARY_COLS,
                                    numeric_cols=numeric_cols,
                                    n_bins=n_bins,
                                ),
                                model_factory(alpha=alpha),
                            ),
                        )
                    )
    return experiments


def build_experiments(batch: str) -> list[Experiment]:
    builders = {
        "baseline_all": build_baseline_experiments,
        "knn_k_refine": build_knn_k_refine_experiments,
        "knn_scaler_compare": build_knn_scaler_compare_experiments,
        "knn_feature_trim": build_knn_feature_trim_experiments,
        "knn_no_name_refine": build_knn_no_name_refine_experiments,
        "knn_no_name_fine": build_knn_no_name_fine_experiments,
        "knn_feature_weights": build_knn_feature_weight_experiments,
        "knn_weight_fine": build_knn_weight_fine_experiments,
        "knn_weight_micro": build_knn_weight_micro_experiments,
        "knn_weight_ablation": build_knn_weight_ablation_experiments,
        "knn_quantile": build_knn_quantile_experiments,
        "nb_lowcard": build_nb_lowcard_experiments,
        "nb_binned_onehot": build_nb_binned_onehot_experiments,
    }
    if batch == "all":
        experiments: list[Experiment] = []
        for builder in builders.values():
            experiments.extend(builder())
        return experiments
    if batch not in builders:
        allowed = ", ".join(["all", *builders.keys()])
        raise ValueError(f"Unknown batch {batch!r}. Allowed batches: {allowed}")
    return builders[batch]()


def evaluate_experiment(
    experiment: Experiment,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, object]:
    scores: list[float] = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        experiment.pipeline.fit(X_train, y_train)
        predictions = experiment.pipeline.predict(X_valid)
        scores.append(accuracy_score(y_valid, predictions))

    return {
        "name": experiment.name,
        "family": experiment.family,
        "batch": experiment.batch,
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "fold_scores": ";".join(f"{score:.6f}" for score in scores),
    }


def evaluate_all(experiments: Iterable[Experiment], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, object]] = []
    for index, experiment in enumerate(experiments, start=1):
        print(f"[{index}] evaluating {experiment.name}")
        rows.append(evaluate_experiment(experiment, X, y, cv))
        latest = rows[-1]
        print(
            f"    mean={latest['cv_mean']:.6f} "
            f"std={latest['cv_std']:.6f}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["cv_mean", "cv_std"],
        ascending=[False, True],
    )
    return results


def load_existing_results() -> pd.DataFrame:
    if not EXPERIMENTS_PATH.exists():
        return pd.DataFrame()
    existing = pd.read_csv(EXPERIMENTS_PATH)
    if "batch" not in existing.columns:
        existing["batch"] = "legacy"
    if "run_at" not in existing.columns:
        existing["run_at"] = ""
    return existing


def merge_results(existing: pd.DataFrame, batch_results: pd.DataFrame) -> pd.DataFrame:
    batch_results = batch_results.copy()
    batch_results["run_at"] = datetime.now().isoformat(timespec="seconds")
    combined = pd.concat([existing, batch_results], ignore_index=True)
    combined["cv_mean"] = pd.to_numeric(combined["cv_mean"], errors="coerce")
    combined["cv_std"] = pd.to_numeric(combined["cv_std"], errors="coerce")
    combined = combined.sort_values(
        ["cv_mean", "cv_std"],
        ascending=[False, True],
    )
    combined = combined.drop_duplicates(subset=["name"], keep="first")
    return combined


def save_best_submission(
    best_experiment: Experiment,
    X: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    best_experiment.pipeline.fit(X, y)
    predictions = best_experiment.pipeline.predict(test).astype(bool)

    submission = sample_submission.copy()
    if not submission["PassengerId"].equals(test["PassengerId"]):
        raise ValueError("sample_submission PassengerId order does not match test.csv")
    submission["Transported"] = predictions
    submission = submission[["PassengerId", "Transported"]]
    submission["Transported"] = submission["Transported"].astype(bool)
    submission.to_csv(BEST_SUBMISSION_PATH, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run allowed KNN/NB experiment batches.")
    parser.add_argument(
        "--batch",
        default="knn_k_refine",
        choices=[
            "baseline_all",
            "knn_k_refine",
            "knn_scaler_compare",
            "knn_feature_trim",
            "knn_no_name_refine",
            "knn_no_name_fine",
            "knn_feature_weights",
            "knn_weight_fine",
            "knn_weight_micro",
            "knn_weight_ablation",
            "knn_quantile",
            "nb_lowcard",
            "nb_binned_onehot",
            "all",
        ],
        help="Small experiment batch to run.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top cumulative CV rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, test, sample_submission = load_data()
    existing_results = load_existing_results()
    previous_best_mean = (
        float(pd.to_numeric(existing_results["cv_mean"], errors="coerce").max())
        if not existing_results.empty
        else float("-inf")
    )

    experiments = build_experiments(args.batch)
    batch_results = evaluate_all(experiments, X, y)
    combined_results = merge_results(existing_results, batch_results)
    combined_results.to_csv(EXPERIMENTS_PATH, index=False)

    best_batch = batch_results.iloc[0]
    best_batch_name = str(best_batch["name"])
    best_batch_mean = float(best_batch["cv_mean"])
    if best_batch_mean > previous_best_mean:
        best_experiment = next(
            experiment for experiment in experiments if experiment.name == best_batch_name
        )
        save_best_submission(best_experiment, X, y, test, sample_submission)
        submission_message = (
            f"Updated best submission because {best_batch_name} improved "
            f"CV mean from {previous_best_mean:.6f} to {best_batch_mean:.6f}."
        )
    else:
        submission_message = (
            f"Kept existing best submission; best batch CV mean "
            f"{best_batch_mean:.6f} did not beat {previous_best_mean:.6f}."
        )

    print(f"\nTop {args.top} cumulative experiments:")
    print(combined_results.head(args.top).to_string(index=False))
    print(f"\nSaved experiments to {EXPERIMENTS_PATH}")
    print(submission_message)
    print(f"Best batch model: {best_batch_name}")


if __name__ == "__main__":
    main()
