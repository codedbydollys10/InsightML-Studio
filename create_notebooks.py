import json
from pathlib import Path

nb_dir = Path("notebooks")
nb_dir.mkdir(exist_ok=True)

notebooks = {
    "01_Data_Understanding.ipynb": [
        {"cell_type": "markdown", "source": "# 01 Data Understanding\nThis notebook explores dataset structure, data types, missing values, duplicates, and initial business insights."},
        {"cell_type": "code", "source": "import pandas as pd\nfrom pathlib import Path\n\ndf = pd.read_csv(Path('data/raw/bank_customer_churn.csv'))\ndf.head()"},
        {"cell_type": "code", "source": "print(f'Rows: {df.shape[0]:,}')\nprint(f'Columns: {df.shape[1]:,}')\nprint(df.dtypes)"},
        {"cell_type": "code", "source": "missing = df.isnull().sum().sort_values(ascending=False)\nprint(missing[missing > 0])\nprint('Duplicate rows:', df.duplicated().sum())"},
        {"cell_type": "code", "source": "display(df.describe(include='all').T.head(15))"},
        {"cell_type": "markdown", "source": "## Key takeaways\n- Dataset size and shape\n- Missing values and duplicate records\n- Candidate target columns and feature types\n- Initial business insights from the raw dataset."},
    ],
    "02_Exploratory_Data_Analysis.ipynb": [
        {"cell_type": "markdown", "source": "# 02 Exploratory Data Analysis\nVisualize distributions, correlations, target balance, and outliers for the dataset."},
        {"cell_type": "code", "source": "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nimport plotly.express as px\nfrom pathlib import Path\n\ndf = pd.read_csv(Path('data/raw/bank_customer_churn.csv'))\nsns.set_style('whitegrid')\ndf.head()"},
        {"cell_type": "code", "source": "print('Target candidates:')\nprint([c for c in df.columns if c.lower() in ['churn', 'target', 'label', 'outcome']])\nprint('Numeric columns:', df.select_dtypes(include='number').columns.tolist())\nprint('Categorical columns:', df.select_dtypes(include='object').columns.tolist())"},
        {"cell_type": "code", "source": "target_col = 'churn' if 'churn' in df.columns else df.columns[-1]\nprint('Using target:', target_col)\nprint(df[target_col].value_counts(normalize=True).mul(100).round(2))"},
        {"cell_type": "code", "source": "numeric = df.select_dtypes(include='number')\nif numeric.shape[1] > 1:\n    corr = numeric.corr()\n    plt.figure(figsize=(10, 8))\n    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu', center=0)\n    plt.title('Numeric feature correlation')\n    plt.show()"},
        {"cell_type": "code", "source": "for col in numeric.columns[:4]:\n    fig = px.histogram(df, x=col, nbins=30, title=f'Distribution of {col}')\n    fig.show()"},
        {"cell_type": "markdown", "source": "## Insights\n- Check imbalance in the selected target.\n- Identify strong correlations and distribution skew.\n- Highlight high-cardinality categorical fields and numeric outliers."},
    ],
    "03_Data_Preprocessing.ipynb": [
        {"cell_type": "markdown", "source": "# 03 Data Preprocessing\nBuild the preprocessing pipeline, handle missing data, encode categories, scale numeric features, and save processed output."},
        {"cell_type": "code", "source": "import pandas as pd\nfrom pathlib import Path\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\n\ndf = pd.read_csv(Path('data/raw/bank_customer_churn.csv'))\ndf.head()"},
        {"cell_type": "code", "source": "df = df.drop_duplicates().copy()\nprint('Rows after deduplication:', len(df))\ncols_to_drop = [c for c in df.columns if df[c].isnull().mean() > 0.6]\nprint('High missing columns to drop:', cols_to_drop)\ndf = df.drop(columns=cols_to_drop)\ndf.isnull().mean().sort_values(ascending=False).head(10)"},
        {"cell_type": "code", "source": "target_col = 'churn' if 'churn' in df.columns else df.columns[-1]\nfeature_cols = [c for c in df.columns if c != target_col]\nnum_cols = df[feature_cols].select_dtypes(include='number').columns.tolist()\ncat_cols = df[feature_cols].select_dtypes(exclude='number').columns.tolist()\nprint('Numeric:', num_cols)\nprint('Categorical:', cat_cols)"},
        {"cell_type": "code", "source": "num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])\ncat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])\npreprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')\nX = preprocessor.fit_transform(df[feature_cols])\nfeature_names = []\nif num_cols:\n    feature_names.extend(num_cols)\nif cat_cols:\n    feature_names.extend(preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols).tolist())\nprint('Processed features:', len(feature_names))\nX_df = pd.DataFrame(X, columns=feature_names)\ny = df[target_col].reset_index(drop=True)\nX_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)\nprint('Train shape:', X_train.shape, 'Test shape:', X_test.shape)"},
        {"cell_type": "code", "source": "from imblearn.over_sampling import SMOTE\nsmote = SMOTE(random_state=42)\nX_train_res, y_train_res = smote.fit_resample(X_train, y_train)\nprint('SMOTE resampled train shape:', X_train_res.shape)\nprint('Class counts after SMOTE:', y_train_res.value_counts())"},
        {"cell_type": "markdown", "source": "## Summary\nThis notebook demonstrates end-to-end preprocessing with deduplication, missing-value handling, encoding, scaling, splitting, and optional SMOTE."},
    ],
    "04_Feature_Engineering.ipynb": [
        {"cell_type": "markdown", "source": "# 04 Feature Engineering\nCreate derived features, apply variance filtering, and inspect PCA on numeric predictors."},
        {"cell_type": "code", "source": "import pandas as pd\nfrom pathlib import Path\nfrom sklearn.feature_selection import VarianceThreshold\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\ndf = pd.read_csv(Path('data/raw/bank_customer_churn.csv'))\ntarget_col = 'churn' if 'churn' in df.columns else df.columns[-1]\ndf.head()"},
        {"cell_type": "code", "source": "df['balance_change'] = df.get('current_balance', 0) - df.get('previous_month_end_balance', 0)\ndf['tenure_years'] = df.get('vintage', 0) / 365\ndf[['balance_change', 'tenure_years']].head()"},
        {"cell_type": "code", "source": "numeric = df.select_dtypes(include='number').drop(columns=[target_col], errors='ignore')\nprint('Numeric feature count before filtering:', numeric.shape[1])\nselector = VarianceThreshold(threshold=0.01)\nselector.fit(numeric.fillna(0))\nselected = numeric.columns[selector.get_support()].tolist()\nprint('Selected numeric features:', len(selected))"},
        {"cell_type": "code", "source": "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(numeric[selected].fillna(0))\npca = PCA(n_components=min(5, X_scaled.shape[1]))\npca_result = pca.fit_transform(X_scaled)\nprint('Explained variance ratio:', pca.explained_variance_ratio_)"},
        {"cell_type": "markdown", "source": "## Business impact\nDerived features and variance selection improve model reliability. PCA helps summarize numeric signal into compact components."},
    ],
    "05_Model_Training.ipynb": [
        {"cell_type": "markdown", "source": "# 05 Model Training\nTrain multiple models for classification or regression and compare performance using a leaderboard."},
        {"cell_type": "code", "source": "import pandas as pd\nfrom pathlib import Path\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import classification_report\n\ndf = pd.read_csv(Path('data/raw/bank_customer_churn.csv'))\ntarget_col = 'churn' if 'churn' in df.columns else df.columns[-1]\nX = df.drop(columns=[target_col])\ny = df[target_col]\nif y.dtype == object or str(y.dtype) == 'string':\n    y = LabelEncoder().fit_transform(y)\nX_train, X_test, y_train, y_test = train_test_split(X.select_dtypes(include='number'), y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)\nclf = RandomForestClassifier(random_state=42, n_jobs=-1)\nclf.fit(X_train, y_train)\ny_pred = clf.predict(X_test)\nprint(classification_report(y_test, y_pred))"},
        {"cell_type": "markdown", "source": "## Note\nThis notebook uses numeric predictors for baseline training. Replace missing values and encode categoricals to extend performance."},
    ],
    "06_Model_Comparison.ipynb": [
        {"cell_type": "markdown", "source": "# 06 Model Comparison\nCompare model performance, inspect leaderboards, and use confusion matrices or residual plots."},
        {"cell_type": "code", "source": "import pandas as pd\nfrom pathlib import Path\nfrom sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score\n\nprint('Load saved model leaderboard and evaluation metrics here.')"},
        {"cell_type": "markdown", "source": "## Analysis\nUse this notebook to compare candidate models after training. Add saved results from the Streamlit app or experiment tracking storage."},
    ],
    "07_Hyperparameter_Tuning.ipynb": [
        {"cell_type": "markdown", "source": "# 07 Hyperparameter Tuning\nTune model hyperparameters using GridSearchCV, RandomizedSearchCV, and Optuna."},
        {"cell_type": "code", "source": "import pandas as pd\nfrom pathlib import Path\nfrom sklearn.model_selection import GridSearchCV\nfrom sklearn.ensemble import RandomForestClassifier\n\ndf = pd.read_csv(Path('data/raw/bank_customer_churn.csv'))\ntarget_col = 'churn' if 'churn' in df.columns else df.columns[-1]\nX = df.drop(columns=[target_col]).select_dtypes(include='number')\ny = df[target_col]\nparam_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}\nclf = RandomForestClassifier(random_state=42, n_jobs=-1)\ngrid = GridSearchCV(clf, param_grid, scoring='f1', cv=3, n_jobs=-1)\ngrid.fit(X, y)\nprint(grid.best_params_)\nprint(grid.best_score_)"},
        {"cell_type": "markdown", "source": "## Next steps\nExtend this notebook with RandomizedSearchCV and Optuna for more efficient hyperparameter exploration."},
    ],
    "08_Explainable_AI.ipynb": [
        {"cell_type": "markdown", "source": "# 08 Explainable AI\nGenerate SHAP and LIME explanations to understand model feature contributions."},
        {"cell_type": "code", "source": "import pandas as pd\nfrom pathlib import Path\n\nprint('Load a trained model and processed data to compute SHAP and LIME explanations.')"},
        {"cell_type": "markdown", "source": "## Guidance\nOnce you have a fitted model, use SHAP to plot summary and waterfall charts, and use LIME for local prediction explanations."},
    ],
    "09_Final_Report.ipynb": [
        {"cell_type": "markdown", "source": "# 09 Final Report\nSummarize dataset quality, model results, and business conclusions in a final deliverable."},
        {"cell_type": "code", "source": "import pandas as pd\nfrom pathlib import Path\n\ndf = pd.read_csv(Path('data/raw/bank_customer_churn.csv'))\nprint('Rows:', len(df))\nprint('Columns:', len(df.columns))\nprint(df.isnull().sum().sort_values(ascending=False).head(10))"},
        {"cell_type": "markdown", "source": "## Report sections\n- Dataset summary\n- EDA highlights\n- Best model and metrics\n- Recommendations and next steps."},
    ],
}

for filename, cells in notebooks.items():
    path = nb_dir / filename
    notebook = {
        'cells': [],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    for cell in cells:
        cell_obj = {'cell_type': cell['cell_type'], 'metadata': {}, 'source': cell['source']}
        if cell['cell_type'] == 'code':
            cell_obj['execution_count'] = None
            cell_obj['outputs'] = []
        notebook['cells'].append(cell_obj)
    path.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    print(f'created {path.name}')
