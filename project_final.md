---

# Sistema de Búsqueda Semántica para el Dataset `melb_data.csv`

## Introducción

Este proyecto tiene como objetivo desarrollar un sistema de búsqueda semántica utilizando embeddings y técnicas de reducción de dimensionalidad. El sistema permite a los usuarios buscar entradas relacionadas en el dataset `melb_data.csv` basándose en la similitud semántica del contenido.

## Objetivos del Proyecto

1. Implementar un sistema de búsqueda semántica utilizando embeddings.
2. Desarrollar una función para la búsqueda y visualización de resultados.
3. Demostrar la comprensión y aplicación de modelos preentrenados (como BERT) y técnicas de embeddings.
4. Presentar y documentar el proyecto de manera clara y concisa.

## Requisitos

Librerías necesarias:
- `gensim`
- `transformers`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`
- `torch`

## Instalación

Para instalar las librerías necesarias, ejecute el siguiente comando:

```bash
pip install gensim transformers scikit-learn matplotlib seaborn pandas numpy torch
```

## Guía de Uso

### Cargar y Descomprimir Datos

Sube el archivo `melb_data.csv.zip` a tu entorno de Google Colab. Luego, descomprime y carga los datos en un DataFrame de pandas.

### Preprocesamiento de Datos

Convierte el texto a minúsculas y realiza limpieza básica para normalizar la información.

### Generación de Embeddings

Utiliza el modelo preentrenado BERT para generar embeddings de las direcciones en el dataset. Este proceso convierte el texto en vectores de alta dimensión que pueden ser comparados entre sí.

### Búsqueda Semántica

Implementa una función de búsqueda que recibe una consulta del usuario y devuelve las entradas más relevantes basadas en la similitud de embeddings.

### Visualización de Resultados

Utiliza técnicas de reducción de dimensionalidad como PCA y t-SNE para visualizar los resultados de búsqueda y crea gráficos con Matplotlib y Seaborn.

### Evaluación

Mide la calidad de las búsquedas utilizando métricas como precisión y recuperación.

## Código del Proyecto

### Cargar y Descomprimir Datos

```python
import zipfile
import pandas as pd

with zipfile.ZipFile('/content/melb_data.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/content')

df = pd.read_csv('/content/melb_data.csv')
print(df.columns)
```

### Preprocesamiento de Datos

```python
def preprocess_text(text):
    text = str(text).lower()
    return text

df['processed_text'] = df['Address'].apply(preprocess_text)
```

### Generación de Embeddings

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_embeddings_batch(texts):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

batch_size = 32
embeddings = []

for i in range(0, len(df), batch_size):
    batch_texts = df['processed_text'][i:i+batch_size].tolist()
    batch_embeddings = get_embeddings_batch(batch_texts)
    embeddings.extend(batch_embeddings)

df['embeddings'] = embeddings
```

### Búsqueda Semántica

```python
from sklearn.metrics.pairwise import cosine_similarity

def search(query, df, top_k=5):
    query_embedding = get_embeddings_batch([preprocess_text(query)])[0]
    df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x]).item())
    results = df.nlargest(top_k, 'similarity')
    return results

query = "turner street"
resultados = search(query, df)
print(resultados[['Suburb', 'Address', 'similarity']])
```

### Visualización de Resultados

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def visualize_embeddings(df):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(np.stack(df['embeddings'].values))
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=df['Regionname'])
    plt.title('Visualización de Embeddings')
    plt.show()

visualize_embeddings(resultados)
```

### Evaluación del Sistema

```python
def precision_at_k(relevant_items, retrieved_items, k):
    relevant_retrieved = [item for item in retrieved_items[:k] if item in relevant_items]
    return len(relevant_retrieved) / k

def recall_at_k(relevant_items, retrieved_items, k):
    relevant_retrieved = [item for item in retrieved_items[:k] if item in relevant_items]
    return len(relevant_retrieved) / len(relevant_items)

relevant_items = df[df['Address'].str.contains('turner', case=False, na=False)]['Address'].tolist()
retrieved_items = resultados['Address'].tolist()

k = 5
precision = precision_at_k(relevant_items, retrieved_items, k)
recall = recall_at_k(relevant_items, retrieved_items, k)

print(f'Precisión en k={k}: {precision:.2f}')
print(f'Recuperación en k={k}: {recall:.2f}')
```

## Video de Presentación

Explicación de la problemática y objetivos del proyecto.
Descripción de los datos utilizados y preprocesamiento.
Demostración de la generación de embeddings.
Ejemplo de búsqueda semántica y visualización de resultados.
Evaluación de la efectividad del sistema.

https://github.com/user-attachments/assets/3853d3f3-7bfb-49a5-902a-5c821c4d31cc

## Recursos y Ejemplos de Proyectos Similares

1. **Hugging Face y FAISS**: Un tutorial detallado sobre cómo usar FAISS para búsqueda semántica con embeddings generados por modelos de transformers está disponible en [Semantic search with FAISS - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter5/6).

2. **DeepSet Haystack**: La librería Haystack proporciona una estructura fácil de usar para implementar sistemas de búsqueda semántica con modelos preentrenados. Más información en [DeepSet](https://www.deepset.ai/blog/how-to-build-a-semantic-search-engine-in-python).

3. **OpenAI API**: Una guía paso a paso para construir un motor de búsqueda semántica utilizando la API de OpenAI. Disponible en [DEV Community](https://dev.to/carolinamonte/introduction-to-semantic-search-with-python-and-openai-api-efg) y [Tucanoo Solutions Ltd.](https://tucanoo.com/semantic-search-tutorial-with-spring-boot-and-openai-embeddings/).

4. **DataCamp y Pinecone**: Un tutorial sobre cómo usar Pinecone y la API de OpenAI para implementar búsqueda semántica. Disponible en [DataCamp](https://www.datacamp.com/tutorial/semantic-search-pinecone-openai).

---

Created by: Th3Mayar (Jose Francisco)
