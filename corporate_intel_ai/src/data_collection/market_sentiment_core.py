#!/usr/bin/env python
"""
Market Sentiment & Behavioral Insights Core Module

This module provides the core functionality for collecting and analyzing
unstructured intelligence from various sources to derive market sentiment
and behavioral insights.

It includes base classes and utilities that are shared across different
source-specific analyzers.
"""

import os
import json
import logging
import hashlib
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from abc import ABC, abstractmethod
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from textblob import TextBlob

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Ensure NLTK resources are available
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

# Try to load spaCy model, download if not available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading spaCy model. This may take a moment...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class SentimentAnalyzer:
    """
    Utility class for performing sentiment analysis on text.
    Provides multiple analysis methods and aggregated results.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer with various NLP tools."""
        self.vader = SentimentIntensityAnalyzer()
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with compound, positive, negative, and neutral scores
        """
        return self.vader.polarity_scores(text)
    
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity (-1 to 1) and subjectivity (0 to 1) scores
        """
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
    
    def analyze_spacy(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using spaCy for entity recognition and other insights.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with entities and other linguistic features
        """
        doc = nlp(text)
        
        # Extract entities
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents
        ]
        
        # Extract key phrases (noun chunks)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        
        return {
            "entities": entities,
            "noun_chunks": noun_chunks,
            "tokens": len(doc),
            "sentences": len(list(doc.sents))
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis using multiple methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with combined sentiment analysis results
        """
        if not text or not isinstance(text, str):
            return {
                "error": "Invalid text input",
                "vader": {},
                "textblob": {},
                "spacy": {},
                "summary": {
                    "sentiment": "neutral",
                    "confidence": 0.0
                }
            }
        
        # Get results from different analyzers
        vader_results = self.analyze_vader(text)
        textblob_results = self.analyze_textblob(text)
        spacy_results = self.analyze_spacy(text)
        
        # Determine overall sentiment
        compound_score = vader_results.get("compound", 0)
        polarity_score = textblob_results.get("polarity", 0)
        
        # Weighted average of both scores
        combined_score = (compound_score * 0.7) + (polarity_score * 0.3)
        
        # Determine sentiment label and confidence
        if combined_score >= 0.05:
            sentiment = "positive"
            confidence = min(abs(combined_score) * 5, 1.0)  # Scale to 0-1
        elif combined_score <= -0.05:
            sentiment = "negative"
            confidence = min(abs(combined_score) * 5, 1.0)  # Scale to 0-1
        else:
            sentiment = "neutral"
            confidence = 1.0 - (min(abs(combined_score) * 10, 0.5))  # Higher confidence for scores closer to 0
        
        return {
            "vader": vader_results,
            "textblob": textblob_results,
            "spacy": spacy_results,
            "summary": {
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "combined_score": round(combined_score, 2)
            }
        }
    
    def analyze_document_sections(self, sections: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze sentiment for different sections of a document.
        
        Args:
            sections: Dictionary mapping section names to text content
            
        Returns:
            Dictionary mapping section names to sentiment analysis results
        """
        results = {}
        
        for section_name, section_text in sections.items():
            results[section_name] = self.analyze_text(section_text)
        
        return results
    
    def analyze_time_series(self, texts: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a series of texts over time.
        
        Args:
            texts: List of (timestamp, text) tuples
            
        Returns:
            List of dictionaries with timestamp and sentiment analysis results
        """
        results = []
        
        for timestamp, text in texts:
            analysis = self.analyze_text(text)
            results.append({
                "timestamp": timestamp,
                "analysis": analysis
            })
        
        return results


class BehavioralInsightsExtractor:
    """
    Utility class for extracting behavioral insights from text.
    Focuses on identifying patterns that indicate strategic behavior.
    """
    
    def __init__(self):
        """Initialize the behavioral insights extractor."""
        # Load spaCy for NLP processing
        self.nlp = nlp
        
        # Hedging phrases indicate uncertainty or hesitation
        self.hedging_phrases = [
            "sort of", "kind of", "somewhat", "in a way", "to some extent",
            "more or less", "rather", "quite", "pretty much", "fairly",
            "basically", "generally", "usually", "typically", "often",
            "sometimes", "occasionally", "rarely", "seldom", "hardly",
            "scarcely", "barely", "almost", "nearly", "virtually",
            "approximately", "about", "around", "roughly", "more or less",
            "we believe", "we think", "we expect", "we anticipate",
            "we estimate", "we project", "we forecast", "we predict",
            "we assume", "we presume", "we suppose", "we suspect",
            "we guess", "we reckon", "we consider", "we deem",
            "we regard", "we view", "we perceive", "we conceive",
            "we judge", "we assess", "we evaluate", "we appraise",
            "we rate", "we rank", "we grade", "we score",
            "we measure", "we gauge", "we estimate", "we calculate",
            "we compute", "we determine", "we ascertain", "we establish",
            "possibly", "perhaps", "maybe", "conceivably", "potentially",
            "presumably", "supposedly", "allegedly", "reputedly", "ostensibly",
            "apparently", "seemingly", "evidently", "manifestly", "patently",
            "clearly", "obviously", "plainly", "undoubtedly", "indubitably",
            "unquestionably", "undeniably", "incontrovertibly", "incontestably",
            "irrefutably", "indisputably", "unarguably", "inarguably"
        ]
        
        # Confidence phrases indicate certainty or conviction
        self.confidence_phrases = [
            "definitely", "certainly", "absolutely", "positively", "undoubtedly",
            "unquestionably", "indisputably", "incontrovertibly", "irrefutably",
            "without a doubt", "beyond doubt", "beyond question", "beyond dispute",
            "beyond controversy", "beyond contention", "beyond argument",
            "we are confident", "we are certain", "we are sure", "we are positive",
            "we are convinced", "we are persuaded", "we are satisfied",
            "we are assured", "we are guaranteed", "we are promised",
            "we know", "we understand", "we recognize", "we acknowledge",
            "we accept", "we admit", "we concede", "we grant",
            "we will", "we shall", "we must", "we should", "we ought to",
            "we need to", "we have to", "we are going to", "we plan to",
            "we intend to", "we aim to", "we hope to", "we expect to",
            "we anticipate", "we project", "we forecast", "we predict",
            "clearly", "obviously", "plainly", "evidently", "manifestly",
            "patently", "transparently", "visibly", "noticeably", "markedly",
            "strikingly", "glaringly", "blatantly", "flagrantly", "egregiously"
        ]
        
        # Evasion phrases indicate avoiding direct answers
        self.evasion_phrases = [
            "as I said before", "as I mentioned earlier", "as I stated previously",
            "as we discussed", "as we talked about", "as we covered",
            "moving on", "let's move on", "turning to", "let's turn to",
            "shifting focus", "let's shift focus", "changing topics",
            "let's change topics", "on another note", "on a different note",
            "in other news", "in other matters", "in other respects",
            "regarding other issues", "concerning other matters",
            "with respect to other topics", "as for other subjects",
            "I'd rather not comment", "I'd prefer not to comment",
            "I'd rather not discuss", "I'd prefer not to discuss",
            "I'd rather not speculate", "I'd prefer not to speculate",
            "I'd rather not get into", "I'd prefer not to get into",
            "I'm not at liberty to", "I'm not at liberty to discuss",
            "I'm not at liberty to comment", "I'm not at liberty to speculate",
            "I'm not authorized to", "I'm not authorized to discuss",
            "I'm not authorized to comment", "I'm not authorized to speculate",
            "I'm not in a position to", "I'm not in a position to discuss",
            "I'm not in a position to comment", "I'm not in a position to speculate"
        ]
        
        # Nervousness indicators in text
        self.nervousness_indicators = [
            "um", "uh", "er", "ah", "like", "you know", "I mean",
            "sort of", "kind of", "I guess", "I suppose", "well",
            "actually", "basically", "literally", "honestly", "frankly",
            "to be honest", "to be frank", "to tell the truth",
            "to be perfectly honest", "to be perfectly frank",
            "to be perfectly truthful", "to be completely honest",
            "to be completely frank", "to be completely truthful"
        ]
    
    def extract_hedging(self, text: str) -> Dict[str, Any]:
        """
        Extract hedging language from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with hedging phrases found and their count
        """
        text_lower = text.lower()
        found_phrases = []
        
        for phrase in self.hedging_phrases:
            if phrase in text_lower:
                found_phrases.append(phrase)
        
        return {
            "hedging_phrases": found_phrases,
            "hedging_count": len(found_phrases),
            "hedging_ratio": len(found_phrases) / max(len(text.split()), 1)
        }
    
    def extract_confidence(self, text: str) -> Dict[str, Any]:
        """
        Extract confidence language from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with confidence phrases found and their count
        """
        text_lower = text.lower()
        found_phrases = []
        
        for phrase in self.confidence_phrases:
            if phrase in text_lower:
                found_phrases.append(phrase)
        
        return {
            "confidence_phrases": found_phrases,
            "confidence_count": len(found_phrases),
            "confidence_ratio": len(found_phrases) / max(len(text.split()), 1)
        }
    
    def extract_evasion(self, text: str) -> Dict[str, Any]:
        """
        Extract evasive language from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with evasion phrases found and their count
        """
        text_lower = text.lower()
        found_phrases = []
        
        for phrase in self.evasion_phrases:
            if phrase in text_lower:
                found_phrases.append(phrase)
        
        return {
            "evasion_phrases": found_phrases,
            "evasion_count": len(found_phrases),
            "evasion_ratio": len(found_phrases) / max(len(text.split()), 1)
        }
    
    def extract_nervousness(self, text: str) -> Dict[str, Any]:
        """
        Extract indicators of nervousness from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with nervousness indicators found and their count
        """
        text_lower = text.lower()
        found_indicators = []
        
        for indicator in self.nervousness_indicators:
            if indicator in text_lower:
                found_indicators.append(indicator)
        
        return {
            "nervousness_indicators": found_indicators,
            "nervousness_count": len(found_indicators),
            "nervousness_ratio": len(found_indicators) / max(len(text.split()), 1)
        }
    
    def analyze_behavioral_patterns(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive behavioral analysis on text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with combined behavioral analysis results
        """
        if not text or not isinstance(text, str):
            return {
                "error": "Invalid text input",
                "hedging": {},
                "confidence": {},
                "evasion": {},
                "nervousness": {},
                "summary": {
                    "behavioral_assessment": "neutral",
                    "confidence": 0.0
                }
            }
        
        # Get results from different extractors
        hedging_results = self.extract_hedging(text)
        confidence_results = self.extract_confidence(text)
        evasion_results = self.extract_evasion(text)
        nervousness_results = self.extract_nervousness(text)
        
        # Calculate behavioral scores
        hedging_score = hedging_results.get("hedging_ratio", 0) * 10  # Scale up for comparison
        confidence_score = confidence_results.get("confidence_ratio", 0) * 10
        evasion_score = evasion_results.get("evasion_ratio", 0) * 10
        nervousness_score = nervousness_results.get("nervousness_ratio", 0) * 10
        
        # Calculate transparency score (inverse of evasion)
        transparency_score = 1.0 - evasion_score if evasion_score <= 1.0 else 0.0
        
        # Calculate certainty score (confidence minus hedging)
        certainty_score = confidence_score - hedging_score
        
        # Calculate composure score (inverse of nervousness)
        composure_score = 1.0 - nervousness_score if nervousness_score <= 1.0 else 0.0
        
        # Determine overall behavioral assessment
        if certainty_score > 0.2 and transparency_score > 0.7 and composure_score > 0.7:
            assessment = "confident_and_transparent"
            assessment_confidence = min((certainty_score + transparency_score + composure_score) / 3, 1.0)
        elif certainty_score < -0.2 and transparency_score < 0.3:
            assessment = "evasive_and_uncertain"
            assessment_confidence = min((abs(certainty_score) + (1 - transparency_score)) / 2, 1.0)
        elif certainty_score > 0.2 and transparency_score < 0.3:
            assessment = "confident_but_evasive"
            assessment_confidence = min((certainty_score + (1 - transparency_score)) / 2, 1.0)
        elif certainty_score < -0.2 and transparency_score > 0.7:
            assessment = "transparent_but_uncertain"
            assessment_confidence = min((abs(certainty_score) + transparency_score) / 2, 1.0)
        elif composure_score < 0.3:
            assessment = "nervous"
            assessment_confidence = min(1 - composure_score, 1.0)
        else:
            assessment = "neutral"
            assessment_confidence = 0.5
        
        return {
            "hedging": hedging_results,
            "confidence": confidence_results,
            "evasion": evasion_results,
            "nervousness": nervousness_results,
            "scores": {
                "certainty": round(certainty_score, 2),
                "transparency": round(transparency_score, 2),
                "composure": round(composure_score, 2)
            },
            "summary": {
                "behavioral_assessment": assessment,
                "confidence": round(assessment_confidence, 2)
            }
        }


class MarketSentimentCollector(ABC):
    """
    Abstract base class for all market sentiment collectors.
    Provides common functionality for caching, data handling, and analysis.
    """
    
    def __init__(self, output_dir: str = "data/market_sentiment", use_cache: bool = True, 
                 cache_expiry_days: int = 7):
        """
        Initialize the market sentiment collector.
        
        Args:
            output_dir: Directory to store output data
            use_cache: Whether to use caching
            cache_expiry_days: Number of days before cache expires
        """
        self.output_dir = output_dir
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        self.cache_dir = os.path.join(output_dir, "_cache")
        
        # Create output and cache directories
        os.makedirs(output_dir, exist_ok=True)
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer()
        self.behavioral_extractor = BehavioralInsightsExtractor()
        
        # Set up session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _get_cache_key(self, prefix: str, query_str: str) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            prefix: Prefix for the cache key
            query_str: Query string to hash
            
        Returns:
            Cache key string
        """
        hash_obj = hashlib.md5(query_str.encode('utf-8'))
        return f"{prefix}_{hash_obj.hexdigest()}"
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get data from cache if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data if available, None otherwise
        """
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        # Check if cache is expired
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.now() - cache_time).days > self.cache_expiry_days:
            logger.info(f"Cache expired for {cache_key}")
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Any) -> bool:
        """
        Save data to cache.
        
        Args:
            cache_key: Cache key
            data: Data to cache
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_cache:
            return False
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment and behavioral patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment and behavioral analysis results
        """
        sentiment_results = self.sentiment_analyzer.analyze_text(text)
        behavioral_results = self.behavioral_extractor.analyze_behavioral_patterns(text)
        
        return {
            "sentiment": sentiment_results,
            "behavioral": behavioral_results,
            "text_length": len(text),
            "word_count": len(text.split())
        }
    
    def export_to_csv(self, data: List[Dict], output_file: str) -> str:
        """
        Export sentiment data to CSV.
        
        Args:
            data: List of dictionaries containing sentiment data
            output_file: Path to output CSV file
            
        Returns:
            Path to the created CSV file
        """
        logger.info(f"Exporting sentiment data to CSV: {output_file}")
        
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write to CSV
            df.to_csv(output_file, index=False)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return ""
    
    def export_to_excel(self, data_dict: Dict[str, List[Dict]], output_file: str) -> str:
        """
        Export sentiment data to Excel with multiple sheets.
        
        Args:
            data_dict: Dictionary mapping sheet names to lists of dictionaries
            output_file: Path to output Excel file
            
        Returns:
            Path to the created Excel file
        """
        logger.info(f"Exporting sentiment data to Excel: {output_file}")
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Create Excel writer
            with pd.ExcelWriter(output_file) as writer:
                for sheet_name, data in data_dict.items():
                    # Convert data to DataFrame
                    df = pd.DataFrame(data)
                    
                    # Write to Excel sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return ""
    
    @abstractmethod
    def collect_data(self, *args, **kwargs) -> List[Dict]:
        """
        Collect sentiment data from the source.
        To be implemented by subclasses.
        
        Returns:
            List of dictionaries containing sentiment data
        """
        pass
    
    @abstractmethod
    def analyze_data(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze collected sentiment data.
        To be implemented by subclasses.
        
        Args:
            data: List of dictionaries containing sentiment data
            
        Returns:
            Dictionary with analysis results
        """
        pass
