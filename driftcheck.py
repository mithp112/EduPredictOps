from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset



def check_concept_drift1(school_name, type_train, list_year, num_features=54):
    drift_matrix = pd.DataFrame(index=list_year, columns=list_year)
    correlation_vectors = {}

    for year in list_year:
        file_path = os.path.join('data', school_name, f"{type_train}_20{year}_20{year+3}.xlsx")
        df = pd.read_excel(file_path)

        x = df.iloc[:, :num_features]
        y = df.iloc[:, num_features:]

        # Tính ma trận tương quan giữa từng cặp (x, y)
        corr_matrix = pd.DataFrame(index=x.columns, columns=y.columns)
        for x_col in x.columns:
            for y_col in y.columns:
                corr_matrix.loc[x_col, y_col] = x[x_col].corr(y[y_col])

        correlation_vectors[year] = corr_matrix.astype(float).values.flatten()

    # So sánh giữa các năm bằng khoảng cách cosine (1 - similarity)
    for y1 in list_year:
        for y2 in list_year:
            vec1 = correlation_vectors[y1].reshape(1, -1)
            vec2 = correlation_vectors[y2].reshape(1, -1)
            drift_score = 1 - cosine_similarity(vec1, vec2)[0][0]
            drift_matrix.loc[y1, y2] = round(drift_score, 4)
    show_heatmap = True
    if show_heatmap:
        plt.figure(figsize=(10, 6))
        sns.heatmap(drift_matrix.astype(float), annot=True, fmt=".3f", cmap="coolwarm")
        plt.title(f"Concept Drift giữa các năm - {school_name} ({type_train})")
        plt.show()

    return drift_matrix


def check_concept_drift2(school_name, type_train, list_year, num_features=54, output_dir="drift_reports"):
    os.makedirs(output_dir, exist_ok=True)

    for y1, y2 in combinations(list_year, 2):
        file1 = os.path.join('data', school_name, f"{type_train}_20{y1}_20{y1+3}.xlsx")
        file2 = os.path.join('data', school_name, f"{type_train}_20{y2}_20{y2+3}.xlsx")


        df1 = pd.read_excel(file1).iloc[:, :num_features]
        df2 = pd.read_excel(file2).iloc[:, :num_features]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df1, current_data=df2)

        filename = f"{school_name}_{type_train}_{y1}_vs_{y2}.html"
        report.save_html(os.path.join(output_dir, filename))

def check_data_drift(school_name, type_train, list_year, num_features, output_dir="drift_reports/data_drift"):
    os.makedirs(output_dir, exist_ok=True)

    for y1, y2 in combinations(list_year, 2):
        file1 = os.path.join('data', school_name, f"{type_train}_{y1}_{y1+3}.xlsx")
        file2 = os.path.join('data', school_name, f"{type_train}_{y2}_{y2+3}.xlsx")

        df1 = pd.read_excel(file1).iloc[:, :num_features]
        df2 = pd.read_excel(file2).iloc[:, :num_features]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df1, current_data=df2)

        report_path = os.path.join(output_dir, f"{school_name}_{type_train}_datadrift_{y1}_vs_{y2}.html")
        report.save_html(report_path)


