import re
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

EMOCOES = {
    1: 'Neutro',
    2: 'Felicidade',
    3: 'Raiva',
    4: 'Surpresa',
    5: 'Tristeza',
    6: 'Disgosto',
    7: 'Medo'
}

TRAIN = 'dialogo_friends_20000.csv'
TEST = 'tokyo_2020_tweets.csv'

def load_data(name: str) -> pd.DataFrame:
    df = pd.read_csv(name)
    df = df.sample(100, random_state=42)

    return df

def preprocess_data(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=[text_column])
    df["texto_limpo"] = df[text_column].apply(limpar_texto)
    df = df[df["texto_limpo"].str.strip() != ""]

    return df

def limpar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return "N/A"

    texto = texto.lower()

    texto = re.sub(r"http\\S+", "", texto)
    texto = re.sub(r"@[A-Za-z0-9_]+", "", texto)
    texto = re.sub(r"[^a-zà-ÿ\s]", "", texto)

    return texto

def generate_wordcloud(texts):
    text = " ".join(texts.astype(str))

    if text.strip() == "":
        print("Nenhum texto encontrado.")
        return

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def create_tfidf_features(texts, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english'
    )

    x = vectorizer.fit_transform(texts)

    return x, vectorizer

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    print("=" * 50)
    print("ACURÁCIA")
    print("=" * 50)
    print(f"{accuracy:.4f}")

    print("\n" + "=" * 50)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 50)

    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    disp.plot(
        cmap="Blues",
        ax=ax
    )

    plt.title("Matriz de Confusão")

    plt.show()

def train_logistic_regression(X, y):
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return model, x_test, y_test, y_pred

def show_top_features(vectorizer, n=20):
    features = vectorizer.get_feature_names_out()

    print(features[:n])


def train_random_forest(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    return model, x_test, y_test, y_pred

if __name__ == '__main__':
    pass