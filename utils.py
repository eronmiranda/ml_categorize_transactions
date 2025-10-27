from pathlib import Path
import pandas as pd  # pylint: disable=E0401
from sklearn.model_selection import train_test_split  # pylint: disable=E0401
from sklearn.feature_extraction.text import TfidfVectorizer  # pylint: disable=E0401
from sklearn.linear_model import LogisticRegression  # pylint: disable=E0401
from sklearn.metrics import classification_report  # pylint: disable=E0401
import joblib


def import_data(dir_name: str) -> pd.DataFrame:
    """Import data from CSV files in the specified directory"""
    base_dir = Path(__file__).resolve().parent
    csv_dir = base_dir / dir_name

    if csv_dir.exists() and csv_dir.is_dir():
        files = sorted(csv_dir.glob('*.csv'))
    else:
        files = []

    frames = []
    for f in files:
        print(f"Processing {f}")
        frames.append(pd.read_csv(f))

    if frames:
        data_frame = pd.concat(frames, ignore_index=True)
    else:
        data_frame = pd.DataFrame()

    return data_frame


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the transaction data"""
    print("Cleaning the data...")
    data['Transaction'] = data['Transaction'].str.replace(
        r'\s{2,}.*|\\t.*', '', regex=True
    ).str.replace(
        r'\*', '', regex=True  # Remove asterisks
    ).str.replace(
        r'(?<!\w)\d+(?!\w)', '', regex=True  # Remove standalone numbers
    ).str.replace(
        r'[*#\-=]', '', regex=True  # Remove special characters
    ).str.replace(
        # Remove city names
        r'\s+(Edmonton|Calgary|Canada|CA|CAN)$', '', regex=True, case=False
    ).str.replace(
        r'\s+', ' ', regex=True
    ).str.strip().str.upper()

    new_data = data.drop(columns=['Card Member', 'Account Number', 'Business Spend',
                                  'Unnamed: 10', 'Date', 'Month-Billed', 'Charges $', 'Credits $'])

    return new_data


def prepare_data(data: pd.DataFrame) -> tuple:
    """Prepare data by combining features and splitting into train/test sets"""
    x = data['Category'] + ' ' + data['Transaction']
    y = data['Sub-Category']
    return train_test_split(x, y, test_size=0.2)


def create_and_train_model(x_train, y_train, max_features=5000) -> tuple:
    """Create and train the model with TF-IDF vectorization"""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    x_train_tfidf = vectorizer.fit_transform(x_train)

    model = LogisticRegression(max_iter=200, class_weight='balanced')
    model.fit(x_train_tfidf, y_train)

    return model, vectorizer


def evaluate_model(model, vectorizer, x_test, y_test) -> str:
    """Evaluate the model and print classification report"""
    x_test_tfidf = vectorizer.transform(x_test)
    predictions = model.predict(x_test_tfidf)

    return classification_report(y_test, predictions)


def save_model(model, vectorizer, model_path: str, vectorizer_path: str) -> None:
    """Save the trained model and vectorizer"""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


def load_model(model_path: str, vectorizer_path: str) -> tuple:
    """Load the trained model and vectorizer from disk"""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
