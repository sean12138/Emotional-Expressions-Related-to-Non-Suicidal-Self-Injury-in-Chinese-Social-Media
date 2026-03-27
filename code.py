import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.utils import resample

class NSSIAnalysisEngine:
  def __init__(self, random_seed=24):
  """
        Initializes the engine with the fixed random seed for reproducibility[cite: 145].
        """
self.seed = random_seed
self.emotions = ['sadness', 'fear', 'anger', 'disgust', 'happiness', 'acceptance', 'surprise']
self.lexicon = {} # To be loaded via load_lexicon()

def preprocess(self, text):
  """
        Stages 1-4: Data Cleaning and Tokenization [cite: 409-415, 458-465].
        Includes removal of URLs, HTML, and length filtering (>=12 chars)[cite: 73, 459].
        """
if not isinstance(text, str) or len(text) < 12:
  return None

# Clean URLs and HTML tags [cite: 459, 460]
text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
# Normalize punctuation [cite: 461]
text = re.sub(r'[^\w\s]', '', text)
# Tokenize and remove stopwords [cite: 463, 464]
tokens = [t for t in jieba.lcut(text) if len(t) > 1]
return tokens

def classify_sentiment(self, tokens):
  """
        Stage 6: Lexicon-based Sentiment Scoring [cite: 91, 466-473].
        Calculates weights and assigns labels based on the hybrid lexicon[cite: 95, 424].
        """
if not tokens:
  return None

# Initialize emotion score vector [cite: 467]
score_vector = np.zeros(len(self.emotions))

for t in tokens:
  if t in self.lexicon:
  # Retrieve category index and weight [cite: 470, 471]
  idx = self.lexicon[t]['index']
weight = self.lexicon[t]['weight']
score_vector[idx] += weight

# Assign emotion via argmax [cite: 473]
if np.max(score_vector) == 0:
  return None
return self.emotions[np.argmax(score_vector)]

def extract_thematic_clusters(self, corpus):
  """
        Section 2.5: K-Means Clustering for Latent Thematic Structures [cite: 131-135].
        Uses TF-IDF unigram-bigram and SVD reduction to 120 components[cite: 132, 133].
        """
# Vectorization [cite: 132]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.92, min_df=12)
X = vectorizer.fit_transform(corpus)

# Dimensionality Reduction (Truncated SVD) [cite: 132, 133]
svd = TruncatedSVD(n_components=120, random_state=self.seed)
X_reduced = svd.fit_transform(X)

# K-Means (k=7) [cite: 134]
kmeans = KMeans(n_clusters=7, max_iter=300, random_state=self.seed)
labels = kmeans.fit_predict(X_reduced)

# Internal Validity Indices [cite: 398, 439]
metrics = {
  "silhouette": silhouette_score(X_reduced, labels),
  "calinski_harabasz": calinski_harabasz_score(X_reduced, labels),
  "davies_bouldin": davies_bouldin_score(X_reduced, labels)
}
return labels, metrics

def run_bootstrap_validation(self, data, n_iterations=1000):
  """
        S2.4: Stability Testing via Bootstrap Resampling[cite: 373, 385, 498].
        """
results = []
for i in range(n_iterations):
  sample = resample(data, random_state=i)
# Logic to re-calculate distribution variance [cite: 386]
results.append(np.mean(sample)) # Simplified for illustration
return np.std(results)

# --- Execution Entry Point ---
if __name__ == "__main__":
  engine = NSSIAnalysisEngine()
print("NSSI Analysis Engine Loaded. Ready for deterministic pipeline execution.")