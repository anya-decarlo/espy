"""
NLP Utilities Module

This module provides natural language processing utilities for the Corporate
Intelligence Automation system, including text preprocessing, entity extraction,
sentiment analysis, and topic modeling.
"""

import logging
import re
import string
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import numpy as np
from collections import Counter

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Configure logging
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Class for preprocessing text data for NLP tasks.
    """
    
    def __init__(self, language: str = "english"):
        """
        Initialize the text preprocessor.
        
        Args:
            language: Language for stopwords and other language-specific processing
        """
        self.language = language
        
        # Download required NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words(language))
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, numbers, and extra whitespace.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, remove_stops: bool = True, lemmatize: bool = True) -> List[str]:
        """
        Preprocess text by cleaning, tokenizing, removing stopwords, and lemmatizing.
        
        Args:
            text: Input text to preprocess
            remove_stops: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens
            
        Returns:
            List of preprocessed tokens
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords if requested
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        return tokens

class EntityExtractor:
    """
    Class for extracting named entities from text.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize the entity extractor.
        
        Args:
            model: spaCy model to use for entity extraction
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            logger.warning(f"Spacy model {model} not found. Please install it with: python -m spacy download {model}")
            self.nlp = None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        if not self.nlp:
            logger.error("spaCy model not loaded. Cannot extract entities.")
            return {}
        
        if not text or not isinstance(text, str):
            return {}
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities by type
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            # Add entity if not already in the list
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def extract_organizations(self, text: str) -> List[str]:
        """
        Extract organization names from text.
        
        Args:
            text: Input text to extract organizations from
            
        Returns:
            List of organization names
        """
        entities = self.extract_entities(text)
        return entities.get("ORG", [])
    
    def extract_people(self, text: str) -> List[str]:
        """
        Extract person names from text.
        
        Args:
            text: Input text to extract person names from
            
        Returns:
            List of person names
        """
        entities = self.extract_entities(text)
        return entities.get("PERSON", [])
    
    def extract_locations(self, text: str) -> List[str]:
        """
        Extract location names from text.
        
        Args:
            text: Input text to extract location names from
            
        Returns:
            List of location names
        """
        entities = self.extract_entities(text)
        return entities.get("GPE", []) + entities.get("LOC", [])

class SentimentAnalyzer:
    """
    Class for analyzing sentiment in text.
    """
    
    def __init__(self, model_name: str = "vader"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the sentiment analysis model to use
        """
        self.model_name = model_name
        
        if model_name == "vader":
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self.analyzer = SentimentIntensityAnalyzer()
            except (ImportError, LookupError):
                nltk.download('vader_lexicon')
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self.analyzer = SentimentIntensityAnalyzer()
        else:
            logger.warning(f"Unsupported sentiment analysis model: {model_name}")
            self.analyzer = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment scores
        """
        if not self.analyzer:
            logger.error("Sentiment analyzer not initialized.")
            return {}
        
        if not text or not isinstance(text, str):
            return {}
        
        if self.model_name == "vader":
            return self.analyzer.polarity_scores(text)
        
        return {}
    
    def classify_sentiment(self, text: str) -> str:
        """
        Classify sentiment in text as positive, negative, or neutral.
        
        Args:
            text: Input text to classify
            
        Returns:
            Sentiment classification (positive, negative, or neutral)
        """
        scores = self.analyze_sentiment(text)
        
        if not scores:
            return "neutral"
        
        compound = scores.get("compound", 0)
        
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    def analyze_text_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment in a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries containing sentiment scores
        """
        return [self.analyze_sentiment(text) for text in texts]

class TopicModeler:
    """
    Class for topic modeling on text data.
    """
    
    def __init__(self, num_topics: int = 5, model_type: str = "lda"):
        """
        Initialize the topic modeler.
        
        Args:
            num_topics: Number of topics to extract
            model_type: Type of topic model to use (lda, nmf)
        """
        self.num_topics = num_topics
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the topic model on a corpus of texts.
        
        Args:
            texts: List of texts to fit the model on
        """
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        
        # Preprocess texts
        processed_texts = [" ".join(self.preprocessor.preprocess(text)) for text in texts]
        
        if self.model_type == "lda":
            from sklearn.decomposition import LatentDirichletAllocation
            
            # Create document-term matrix
            self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
            dtm = self.vectorizer.fit_transform(processed_texts)
            
            # Fit LDA model
            self.model = LatentDirichletAllocation(
                n_components=self.num_topics,
                random_state=42,
                learning_method="online"
            )
            self.model.fit(dtm)
            
        elif self.model_type == "nmf":
            from sklearn.decomposition import NMF
            
            # Create TF-IDF matrix
            self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
            tfidf = self.vectorizer.fit_transform(processed_texts)
            
            # Fit NMF model
            self.model = NMF(
                n_components=self.num_topics,
                random_state=42,
                alpha=0.1,
                l1_ratio=0.5
            )
            self.model.fit(tfidf)
            
        else:
            logger.warning(f"Unsupported topic model type: {self.model_type}")
    
    def get_topics(self, num_words: int = 10) -> List[List[str]]:
        """
        Get the top words for each topic.
        
        Args:
            num_words: Number of words to include for each topic
            
        Returns:
            List of lists, where each inner list contains the top words for a topic
        """
        if not self.model or not self.vectorizer:
            logger.error("Topic model not fitted.")
            return []
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top words for each topic
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_words_idx = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
        
        return topics
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to topic distributions.
        
        Args:
            texts: List of texts to transform
            
        Returns:
            Array of topic distributions for each text
        """
        if not self.model or not self.vectorizer:
            logger.error("Topic model not fitted.")
            return np.array([])
        
        # Preprocess texts
        processed_texts = [" ".join(self.preprocessor.preprocess(text)) for text in texts]
        
        # Transform texts to document-term matrix or TF-IDF matrix
        matrix = self.vectorizer.transform(processed_texts)
        
        # Transform to topic distributions
        return self.model.transform(matrix)
