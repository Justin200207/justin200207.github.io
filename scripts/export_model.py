"""Rebuild the MIE1624 Assignment 2 final estimator and export it for GitHub Pages.

Usage:
    python scripts/export_model.py --csv /path/to/processed_data_assignment2_2026.csv

The dataset is read only with Python. No respondent rows are exported. The website
ships training-derived vocabularies, imputation statistics and learned weights.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

SALARY_ORDER = [
    "$0–<$10,000", "$10,000–<$20,000", "$20,000–<$30,000",
    "$30,000–<$40,000", "$40,000–<$50,000", "$50,000–<$75,000",
    "$75,000–<$100,000", "$100,000–<$125,000", "$125,000–<$150,000",
    "$150,000–<$200,000", "$200,000–<$250,000", "$250,000–<$300,000",
    "$300,000–<$400,000", "$400,000–≤$500,000", ">$500,000",
]
SELECTED_FEATURES = [
    "Country", "Age", "Employment", "EmploymentAddl", "ICorPM",
    "Role", "RoleFunctions", "OrgSize", "WorkSetting", "Education",
    "YearsCode", "WorkExp", "WorkExpSquared", "SOYears", "KnowledgeQ",
    "CloudPlatHaveWorkedWith", "LanguagesHaveWorkedWith", "KnowMgmtHaveWorkedWith",
]
RAW_FEATURES = [c for c in SELECTED_FEATURES if c != "WorkExpSquared"]
RECOMMENDATION_QUESTIONS = [
    "LanguagesHaveWorkedWith", "CloudPlatHaveWorkedWith", "KnowMgmtHaveWorkedWith"
]

class MissingCategoryFiller(BaseEstimator, TransformerMixin):
    def __init__(self, missing_label="__MISSING__"):
        self.missing_label = missing_label

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.astype("string").fillna(self.missing_label).astype(str)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

class MultiSelectEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, delimiter=";", missing_label="__MISSING__"):
        self.delimiter = delimiter
        self.missing_label = missing_label

    def _tokens(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        rows = []
        for values in X.itertuples(index=False, name=None):
            row_tokens = []
            for col, value in zip(self.feature_names_in_, values):
                if pd.isna(value) or not str(value).strip():
                    row_tokens.append(f"{col}={self.missing_label}")
                else:
                    choices = [s.strip() for s in str(value).split(self.delimiter) if s.strip()]
                    row_tokens.extend(f"{col}={choice}" for choice in choices)
            rows.append(row_tokens)
        return rows

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.encoder_ = MultiLabelBinarizer(sparse_output=True)
        self.encoder_.fit(self._tokens(X))
        return self

    def transform(self, X):
        return self.encoder_.transform(self._tokens(X))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.encoder_.classes_, dtype=object)

class CumulativeOrdinalLogit:
    def __init__(self, C=0.3, penalty="l1", max_iter=500, random_state=42):
        self.C, self.penalty, self.max_iter, self.random_state = C, penalty, max_iter, random_state

    def fit(self, X, y):
        self.models_ = []
        y = np.asarray(y, dtype=int)
        for threshold in range(len(SALARY_ORDER) - 1):
            clf = LogisticRegression(C=self.C, penalty=self.penalty,
                                     solver="liblinear", max_iter=self.max_iter,
                                     random_state=self.random_state)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="'penalty' was deprecated", category=FutureWarning)
                warnings.filterwarnings("ignore", message="Inconsistent values: penalty=")
                clf.fit(X, (y <= threshold).astype(int))
            self.models_.append(clf)
        return self

    def predict_proba(self, X):
        cum = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        cum = np.maximum.accumulate(np.clip(cum, 0, 1), axis=1)
        limits = np.column_stack([np.zeros(X.shape[0]), cum, np.ones(X.shape[0])])
        probs = np.maximum(np.diff(limits, axis=1), 0)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        cum = np.cumsum(self.predict_proba(X), axis=1)
        return np.argmax(cum >= 0.5, axis=1)

# Benchmark example is the questionnaire used in the submitted notebook Q5.5.
EXAMPLE = {
    "Country":"Canada", "Age":"18-24 years old", "Employment":"Employed",
    "EmploymentAddl":"Attending school (part-time)", "ICorPM":"Individual contributor",
    "Role":"Developer, desktop or enterprise applications",
    "RoleFunctions":"Debugging or troubleshooting;Reviewing code, pull requests, or technical changes;Writing or maintaining tests;Writing, editing, or generating code",
    "OrgSize":"5,000 to 9,999 employees",
    "WorkSetting":"Hybrid (some in-person, leans heavy to flexibility)",
    "Education":"Bachelor's degree (B.A., B.S., B.Eng., etc.)",
    "YearsCode":6,"WorkExp":2,"SOYears":"I don't use Stack Overflow",
    "KnowledgeQ":"Internal documentation (i.e. Design, product, or planning docs), wiki, or knowledge base;Manager or leadership;Official docs;Past decisions or architecture records;Product, vendor, or API documentation",
    "CloudPlatHaveWorkedWith":"DigitalOcean",
    "LanguagesHaveWorkedWith":"Java;JavaScript",
    "KnowMgmtHaveWorkedWith":"Atlassian Confluence;Microsoft SharePoint"
}


def engineered(raw):
    x = raw.copy()
    x["WorkExpSquared"] = x["WorkExp"] ** 2
    return x[SELECTED_FEATURES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", default=Path(__file__).parents[1] / "model.json", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    assert len(df) == 13507, f"Unexpected dataset size: {len(df)}"
    y = df["ConvertedCompYearlyBucket"].map({v:i for i,v in enumerate(SALARY_ORDER)})
    assert not y.isna().any(), "Unexpected salary bucket category"
    raw = df[RAW_FEATURES].copy()
    train_idx, test_idx = train_test_split(df.index, test_size=.2,
                                           random_state=42, stratify=y)
    train_raw, test_raw = raw.loc[train_idx], raw.loc[test_idx]
    train, test = engineered(train_raw), engineered(test_raw)
    train_y, test_y = y.loc[train_idx].astype(int), y.loc[test_idx].astype(int)
    numeric = [c for c in SELECTED_FEATURES if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in SELECTED_FEATURES if c not in numeric]
    multi = [c for c in categorical if train[c].dropna().astype(str).str.contains(";", regex=False).any()]
    single = [c for c in categorical if c not in multi]
    print('Num:', numeric, 'Single:', single, 'Multi:', multi, flush=True)

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    one_pipe = Pipeline([("fill", MissingCategoryFiller()),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    prep = ColumnTransformer(transformers=[
        ("numerical", num_pipe, numeric),
        ("numeric_missing", MissingIndicator(features="missing-only", error_on_new=False), numeric),
        ("single", one_pipe, single),
        ("multiple", MultiSelectEncoder(), multi),
    ], remainder="drop", sparse_threshold=1.0)
    x_train = prep.fit_transform(train)
    x_test = prep.transform(test)
    print('Encoded shape', x_train.shape, 'type', type(x_train).__name__, flush=True)
    ordinal = CumulativeOrdinalLogit().fit(x_train, train_y)
    preds_train, preds_test = ordinal.predict(x_train), ordinal.predict(x_test)
    test_probs = ordinal.predict_proba(x_test)
    print('Training', dict(MAE=mean_absolute_error(train_y, preds_train),accuracy=accuracy_score(train_y,preds_train),macro_f1=f1_score(train_y,preds_train, average='macro'),within1=np.mean(np.abs(train_y.to_numpy()-preds_train)<=1)))
    print('Testing', dict(MAE=mean_absolute_error(test_y, preds_test),accuracy=accuracy_score(test_y,preds_test),macro_f1=f1_score(test_y,preds_test, average='macro'),within1=np.mean(np.abs(test_y.to_numpy()-preds_test)<=1)),flush=True)
    assert abs(mean_absolute_error(test_y, preds_test) - 1.3179) < 0.00015, "Model differs from submitted report"

    imputer = prep.named_transformers_["numerical"].named_steps["impute"]
    scaler = prep.named_transformers_["numerical"].named_steps["scale"]
    miss_indices = prep.named_transformers_["numeric_missing"].features_
    onehot = prep.named_transformers_["single"].named_steps["onehot"]
    many = prep.named_transformers_["multiple"].encoder_
    offset = 0
    numeric_data = []
    for i,col in enumerate(numeric):
        numeric_data.append({"name":col, "index":offset+i,
                             "median":float(imputer.statistics_[i]),
                             "mean":float(scaler.mean_[i]), "scale":float(scaler.scale_[i])})
    offset += len(numeric)
    missing_data = [{"name":numeric[i],"index":offset+k} for k,i in enumerate(miss_indices)]
    offset += len(miss_indices)
    single_data = []
    for name, cats in zip(single,onehot.categories_):
        categories = [str(c) for c in cats]
        single_data.append({"name":name,"options":categories,
                            "indices":list(range(offset,offset+len(categories)))})
        offset += len(categories)
    multi_data = []
    for col in multi:
        entries = []
        tokens = train_raw[col].dropna().astype(str).str.split(';').explode().str.strip()
        support = tokens.value_counts()
        for i, label in enumerate(many.classes_):
            prefix=col+'='
            if label.startswith(prefix):
                category=label[len(prefix):]
                entries.append({"value":category,"index":offset+i,"support":int(support.get(category,0))})
        multi_data.append({"name":col,"options":entries})
    offset += len(many.classes_)
    assert x_train.shape[1] == offset, (x_train.shape[1],offset)

    model = {
        "title":"Developer Salary Explorer",
        "source":"2026 Stack Overflow Developer Survey / MIE1624 Assignment 2",
        "method":"Cumulative threshold ordinal logistic regression; 14 liblinear L1 models (C=0.3)",
        "labels":SALARY_ORDER,
        "rawFeatures":RAW_FEATURES,
        "numeric":numeric_data,
        "numericMissing":missing_data,
        "single":single_data,
        "multi":multi_data,
        "intercepts":[float(m.intercept_[0]) for m in ordinal.models_],
        "weights":[[float(w) for w in m.coef_[0]] for m in ordinal.models_],
        "recommendationQuestions":RECOMMENDATION_QUESTIONS,
        "minimumSupport":30,
        "totalRespondents":len(df),"trainingRespondents":len(train),"testingRespondents":len(test),
        "metrics":{"testMAE":float(mean_absolute_error(test_y,preds_test)),
                   "testAccuracy":float(accuracy_score(test_y,preds_test)),
                   "testWithinOne":float(np.mean(np.abs(test_y.to_numpy()-preds_test)<=1))},
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(model,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print('Exported:',args.out,'size KB:',round(args.out.stat().st_size/1024),flush=True)

    reference = engineered(pd.DataFrame([EXAMPLE]))
    p_ref = ordinal.predict_proba(prep.transform(reference))[0]
    expected = float(p_ref @ np.arange(len(SALARY_ORDER)))
    print('Notebook example: predicted',int(np.searchsorted(np.cumsum(p_ref),0.5)),
          'peak',np.argmax(p_ref), 'peak probability',max(p_ref), 'expected',expected)
    for skill in [('CloudPlatHaveWorkedWith','Terraform'),('LanguagesHaveWorkedWith','Rust'),('CloudPlatHaveWorkedWith','Homebrew')]:
        x = dict(EXAMPLE)
        x[skill[0]] = ';'.join(sorted(x[skill[0]].split(';')+[skill[1]]))
        new = ordinal.predict_proba(prep.transform(engineered(pd.DataFrame([x]))))[0]
        print(skill, 'delta expected bucket',float((new-p_ref) @ np.arange(15)),flush=True)

if __name__ == "__main__":
    main()
