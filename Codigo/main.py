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

def load_data(name: str, sample_size=None) -> pd.DataFrame:
    try:
        df = pd.read_csv(name, encoding='utf-8', on_bad_lines='skip')

    except Exception as e:
        df = pd.read_csv(name, encoding='latin1', on_bad_lines='skip')

    df = df.dropna(how='all')

    if sample_size is not None:
        sample_size = min(sample_size, len(df))
        df = df.sample(sample_size, random_state=42)

    return df

def classifier_data(df: pd.DataFrame, number_column: str) -> pd.DataFrame:
    df = df.copy()

    extracted = df[number_column].astype(str).str.extract(r'(\d+)')[0]
    df[number_column] = pd.to_numeric(extracted, errors='coerce').astype('Int64')
    df["emocao"] = df[number_column].map(EMOCOES)

    df = df.dropna(subset=["emocao"])

    return df

def limpar_texto_train(texto: str) -> str:
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()
    texto = re.sub(r"enunciado:\s*", "", texto)
    texto = re.sub(r"número:\s*", "", texto)
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@[A-Za-z0-9_]+", "", texto)
    texto = re.sub(r"[^a-zà-ÿ\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def limpar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@[A-Za-z0-9_]+", "", texto)
    texto = re.sub(r"[^a-zà-ÿ\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()

def preprocess_data(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()

    df[text_column] = df[text_column].astype(str)

    df[text_column + "_limpo"] = df[text_column].apply(
        limpar_texto
    )

    return df

def preprocess_data_train(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()

    df[text_column] = df[text_column].astype(str)
    df[text_column + "_limpo"] = df[text_column].apply(limpar_texto_train)

    return df

def generate_wordcloud(texts, title="WordCloud"):
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
    plt.title(title)
    plt.show()

def create_tfidf_features(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
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

    print(classification_report(y_true, y_pred, zero_division=0))

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    disp.plot(cmap="Blues", ax=ax)

    plt.title("Matriz de Confusão")

    plt.show()

def show_top_features(vectorizer, n=20):
    features = vectorizer.get_feature_names_out()

    print("\n" + "=" * 50)
    print("PRINCIPAIS FEATURES")
    print("=" * 50)

    print(features[:n])

def train_logistic_regression(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return model, x_test, y_test, y_pred

def train_random_forest(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return model, x_test, y_test, y_pred

if __name__ == '__main__':
    df_train = load_data(TRAIN, sample_size=5000)
    df_train = df_train.dropna(subset=["enunciado", "número"])
    df_train = preprocess_data_train(df_train, "enunciado")
    df_train = classifier_data(df_train, "número")

    print(df_train.head())

    generate_wordcloud(df_train["enunciado_limpo"], title="WordCloud - Treino")

    print("\nGerando TF-IDF...")

    X, vectorizer = create_tfidf_features(df_train["enunciado_limpo"])
    y = df_train["emocao"]

    lr_model, lr_x_test, lr_y_test, lr_y_pred = train_logistic_regression(X, y)
    evaluate_model(lr_y_test, lr_y_pred)
    plot_confusion_matrix(lr_y_test, lr_y_pred, labels=list(EMOCOES.values()))

    rf_model, rf_x_test, rf_y_test, rf_y_pred = train_random_forest(X, y)
    evaluate_model(rf_y_test, rf_y_pred)
    plot_confusion_matrix(rf_y_test, rf_y_pred, labels=list(EMOCOES.values()))

    show_top_features(vectorizer)

    df_test = load_data(TEST, sample_size=1000)

    if "text" not in df_test.columns:
        raise Exception(
            "A coluna 'text' não existe no dataset de teste."
        )

    df_test = df_test.dropna(subset=["text"])
    df_test = preprocess_data(df_test, "text")

    print(df_test.head())

    textos_teste = df_test["text_limpo"]
    X_novo = vectorizer.transform(textos_teste)

    print("\n" + "=" * 60)
    print("PREVISÕES - REGRESSÃO LOGÍSTICA")
    print("=" * 60)

    previsoes_lr = lr_model.predict(X_novo)
    df_test["emocao_lr"] = previsoes_lr

    print(df_test[["text", "emocao_lr"]].head(10))

    print("\n" + "=" * 60)
    print("PREVISÕES - RANDOM FOREST")
    print("=" * 60)

    previsoes_rf = rf_model.predict(X_novo)
    df_test["emocao_rf"] = previsoes_rf

    print(df_test[["text", "emocao_rf"]].head(10))

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)

    print(df_test[["text", "emocao_lr", "emocao_rf"]].head(20))

    df_test.to_csv("resultado_modelos.csv", index=False)

    print("\nArquivo salvo: resultado_modelos.csv")