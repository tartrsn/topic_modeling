import pandas as pd
import re
import preprocessor as p
from tqdm.notebook import tqdm
import numpy as np
import umap.umap_ as umap
from ast import literal_eval
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import openai
from dotenv import find_dotenv, load_dotenv
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pickle


class ConversationProcessor:
    def __init__(self, input_csv):
        self.input_path = input_csv
        self.df = self._prepare_dataset()

    def _clean_conversations(self, column):
        REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")
        tempArr = [REPLACE_WITH_SPACE.sub(" ", p.clean(line.lower())) for line in column]
        return tempArr

    def _prepare_dataset(self):
        df = pd.read_csv(self.input_path)
        df['clean_conversations'] = self._clean_conversations(df['ConversationRemark'])
        print(df[['ConversationRemark', 'clean_conversations']])
        return df


class EmbeddingsExtractor:
    def __init__(self, df):
        self.df = df

    def _dimensionality_reduction(self, column):
        reducer = umap.UMAP()
        embedding_list = column.tolist()
        embedding_matrix = np.stack(embedding_list, axis=0)
        reduced_embeddings = reducer.fit_transform(embedding_matrix)
        return reduced_embeddings

    def _convert_to_list(self, column):
        return [float(x) for x in column.strip('[]').split()]

    def _embeddings_openai(self):
        client = openai.OpenAI()
        text_values = self.df['clean_conversations'].tolist()
        response = client.embeddings.create(input=text_values, model="text-embedding-ada-002")
        return response.data[0].embedding

    def _embeddings_hug(self, model_name_hug='sentence-transformers/all-mpnet-base-v2'):
        model = SentenceTransformer(model_name_hug)
        text_values = self.df['clean_conversations'].tolist()
        embeddings = model.encode(text_values, show_progress_bar=True)
        r_hug_emb = self._dimensionality_reduction(embeddings)
        return r_hug_emb

    def get_openai_embeddings(self):
        return self._embeddings_openai()

    def get_hug_embeddings(self):
        return self._embeddings_hug()


class ClusterAnalyzer:
    def __init__(self, df):
        self.df = df

    def silhouette_analysis(self, n_clusters_range, silhouette_scores):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(n_clusters_range, silhouette_scores, color='gray')
        max_index = np.argmax(silhouette_scores)
        bars[max_index].set_color('#4d41e0')
        plt.xticks(n_clusters_range)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Determining Optimal Number of Clusters')
        plt.show()

    def _kmeans(self, column, max_n_clusters=20):
        if isinstance(column.loc[0], str):
            column = column.apply(self._convert_to_list)

        X_train = column.to_list()
        n_clusters_range = range(3, max_n_clusters)
        silhouette_scores = []
        kmeans_model = None

        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=1234, verbose=0)
            cluster_labels = kmeans.fit_predict(X_train)
            silhouette_avg = silhouette_score(X_train, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            if silhouette_avg >= max(silhouette_scores):
                kmeans_model = kmeans

        self.silhouette_analysis(n_clusters_range, silhouette_scores)
        return silhouette_scores, kmeans_model

    def analyze_clusters(self, column, method='kmeans'):
        if method == 'kmeans':
            return self._kmeans(column)
        elif method == 'dbscan':
            return self._dbscan(column)
        else:
            raise ValueError("Invalid clustering method")

    def _dbscan(self, column, eps_range=np.arange(0.1, 2.0, 0.1), min_samples_range=range(2, 5)):
        pass


class ClassificationProcessor:
    def __init__(self, df, model):
        self.df = df
        self.model = model

    def openai_classify(self):
        # Description: 2-3 sentences of the common theme among all entries.
        CLASSIFICATION_PROMPT = """
            Given classes with lists of questions and answers. Classify the topic of those questions and answers.
            Return the response in the format:
            Class name: A noun (1-3 words) that encompasses the description and can be used as the class name during classification.
            Do it for all classes.
            """

        def classify_group(prompt):
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": CLASSIFICATION_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2
            )
            return response.choices[0].message.content

        cluster_counts = self.df['cluster_n'].value_counts(sort=False).sort_index()
        query = ""

        for cluster in cluster_counts.index:
            cluster_content = self.df[self.df['cluster_n'] == cluster]['clean_conversations'].values[:30]
            prompt = '\n'.join([f"```{issue}```" for issue in cluster_content])
            query += f"class {cluster + 1}: {prompt}\n"

        print(query)
        cluster_class = classify_group(query)
        print(cluster_class)
        return cluster_class

    def classify_classes(self):
        column = self.df['hug_embeddings']
        if isinstance(column.loc[0], str):
            column = column.apply(self._convert_to_list)

        X_train = column.to_list()
        cluster_labels = self.model.predict(X_train)
        self.df['cluster_n'] = cluster_labels
        classified_classes = self.openai_classify()

        class_matches = re.findall(r'Class (\d+): (.+)', classified_classes)
        class_dict = {int(number)-1: description for number, description in class_matches}
        self.df['openai_descr'] = self.df['cluster_n'].map(class_dict)
        return self.df


class DataExporter:
    def __init__(self, df):
        self.df = df

    def convert_to_excel(self):
        self.df.drop(columns=['clean_conversations', 'hug_embeddings'], axis=1, inplace=True)
        self.df.to_excel(r'output/tweets.xlsx', index=None, header=True)


if __name__ == "__main__":
    # Tweet_clusterization is main script. To run it, enter your directory in input_path. In _prepare_dataset rename the column with customer messages to ConversationRemark.
    input_path = 'your/path'


    # initialize openai client
    load_dotenv(find_dotenv())
    client = openai.OpenAI()

    # Process Conversations
    conversation_processor = ConversationProcessor(input_path)
    df_conversations = conversation_processor.df

    # Extract Embeddings
    embeddings_extractor = EmbeddingsExtractor(df_conversations)
    #openai_embeddings = embeddings_extractor.get_openai_embeddings()
    hug_embeddings = embeddings_extractor.get_hug_embeddings()

    #df_conversations['openai_embeddings'] = openai_embeddings.tolist()
    df_conversations['hug_embeddings'] = hug_embeddings.tolist()

    # Analyze Clusters
    cluster_analyzer = ClusterAnalyzer(df_conversations)
    silhouette_scores, kmeans_model = cluster_analyzer.analyze_clusters(df_conversations['hug_embeddings'])

    # Classify Classes
    classification_processor = ClassificationProcessor(df_conversations, kmeans_model)
    df_classified = classification_processor.classify_classes()

    # Export Data
    data_exporter = DataExporter(df_classified)
    data_exporter.convert_to_excel()
