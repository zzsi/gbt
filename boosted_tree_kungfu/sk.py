"""
sklearn models
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def make_date(df, date_field):
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(
            df[date_field], infer_datetime_format=True, errors="coerce"
        )


def change_string_category(df):
    for n, c in df.items():
        if pd.api.types.is_string_dtype(c):
            df[n] = c.astype("category").cat.as_ordered()


def get_all_dtypes_column(df):
    category_type = []
    int_type = []
    bool_type = []
    for n, c in df.items():
        if isinstance(c.dtype, pd.api.types.CategoricalDtype):
            category_type.append(n)
        if pd.api.types.is_numeric_dtype(c):
            int_type.append(n)
        if pd.api.types.is_bool_dtype(c):
            bool_type.append(n)
    return category_type, int_type, bool_type


def fix_missing_nan_values(df, col, name):
    if pd.api.types.is_numeric_dtype(col):
        if pd.isnull(col).sum():
            df[name + "_na"] = pd.isnull(col)
        df[name] = col.fillna(0)


def convert_cat_codes(df, col, name, max_n_cat):
    if not pd.api.types.is_numeric_dtype(col) and (
        max_n_cat is None or len(col.cat.categories) > max_n_cat
    ):
        df[name] = pd.Categorical(col).codes + 1


def proc_df(
    df,
    y_fld=None,
    skip_flds=None,
    ignore_flds=None,
    do_scale=False,
    na_dict=None,
    preproc_fn=None,
    max_n_cat=None,
    subset=None,
    mapper=None,
):

    if not ignore_flds:
        ignore_flds = []
    if not skip_flds:
        skip_flds = []
    if subset:
        df = get_sample(df, subset)
    else:
        df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn:
        preproc_fn(df)
    if y_fld is None:
        y = None
    else:
        if not pd.api.types.is_numeric_dtype(df[y_fld]):
            df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None:
        na_dict = {}
    else:
        na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n, c in df.items():
        na_dict = fix_missing_nan_values(df, c, n)
    if len(na_dict_initial.keys()) > 0:
        df.drop(
            [
                a + "_na"
                for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))
            ],
            axis=1,
            inplace=True,
        )
    if do_scale:
        mapper = scale_vars(df, mapper)
    for n, c in df.items():
        convert_cat_codes(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale:
        res = res + [mapper]
    return res


class SkTreeModel:
    def __init__(self, boosting_type="rf"):
        self.boosting_type = boosting_type

    def train(self, train_ds, val_ds):
        df_train = train_ds.data
        change_string_category(df_train)
        y_fld = None  # "SalePrice"
        X_train, y_train, nas = proc_df(df_train, y_fld)
        y_train = train_ds.label
        m = RandomForestRegressor(n_jobs=-1)
        m.fit(X_train, y_train)
        self.model = m

    def predict(self, val_ds):
        df_val = val_ds.data
        change_string_category(df_val)
        # TODO: should not do proc_df again, but to apply the same proc_df parameters as in train.
        X_val, y_val, nas = proc_df(df_val, None)
        return self.model.predict(X_val)
