#!/usr/bin/env python
"""
Earnings Call Sentiment Analyzer

This module analyzes earnings call transcripts and audio to extract sentiment,
behavioral insights, and key indicators of executive confidence or uncertainty.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import speech_recognition as sr
from pydub import AudioSegment

# Import from core module
from corporate_intel_ai.src.data_collection.market_sentiment_core import (
    MarketSentimentCollector, SentimentAnalyzer, BehavioralInsightsExtractor
)

# Configure logging
logger = logging.getLogger(__name__)

class EarningsCallSentimentAnalyzer(MarketSentimentCollector):
    """
    Analyzes earnings call transcripts and audio to extract sentiment,
    behavioral insights, and key indicators of executive confidence or uncertainty.
    """
    
    def __init__(self, output_dir: str = "data/earnings_call_sentiment", 
                 use_cache: bool = True, cache_expiry_days: int = 30,
                 api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the earnings call sentiment analyzer.
        
        Args:
            output_dir: Directory to store output data
            use_cache: Whether to use caching
            cache_expiry_days: Number of days before cache expires
            api_keys: Dictionary of API keys for various services
        """
        super().__init__(output_dir, use_cache, cache_expiry_days)
        self.api_keys = api_keys or {}
        
        # Sources for earnings call transcripts
        self.sources = {
            "seeking_alpha": "https://seekingalpha.com/earnings/earnings-call-transcripts",
            "motley_fool": "https://www.fool.com/earnings-call-transcripts",
            "yahoo_finance": "https://finance.yahoo.com/calendar/earnings"
        }
        
        # Sections of interest in earnings calls
        self.sections_of_interest = [
            "opening_remarks",
            "financial_results",
            "guidance",
            "qa_session"
        ]
        
        # Key executives to track
        self.key_executives = [
            "CEO", "Chief Executive Officer",
            "CFO", "Chief Financial Officer",
            "COO", "Chief Operating Officer",
            "CTO", "Chief Technology Officer",
            "President", "Chairman"
        ]
        
        # Initialize speech recognition if needed
        self.recognizer = sr.Recognizer()
    
    def get_earnings_call_transcript(self, ticker: str, quarter: Optional[str] = None, 
                                    year: Optional[int] = None) -> Dict[str, Any]:
        """
        Get earnings call transcript for a company.
        
        Args:
            ticker: Company ticker symbol
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year
            
        Returns:
            Dictionary containing transcript data
        """
        logger.info(f"Getting earnings call transcript for {ticker} {quarter} {year}")
        
        # Set default values if not provided
        if not year:
            year = datetime.now().year
        if not quarter:
            current_month = datetime.now().month
            if current_month <= 3:
                quarter = "Q1"
            elif current_month <= 6:
                quarter = "Q2"
            elif current_month <= 9:
                quarter = "Q3"
            else:
                quarter = "Q4"
        
        # Create a cache key
        cache_key = self._get_cache_key("earnings_transcript", f"{ticker}_{quarter}_{year}")
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached transcript for {ticker} {quarter} {year}")
            return cached_result
        
        # Try different sources
        transcript = None
        
        # Try Seeking Alpha
        try:
            transcript = self._get_transcript_from_seeking_alpha(ticker, quarter, year)
        except Exception as e:
            logger.warning(f"Error getting transcript from Seeking Alpha: {e}")
        
        # Try Motley Fool if Seeking Alpha failed
        if not transcript:
            try:
                transcript = self._get_transcript_from_motley_fool(ticker, quarter, year)
            except Exception as e:
                logger.warning(f"Error getting transcript from Motley Fool: {e}")
        
        # If all sources failed, return empty result
        if not transcript:
            logger.warning(f"Could not find transcript for {ticker} {quarter} {year}")
            return {
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "found": False,
                "error": "Transcript not found"
            }
        
        # Cache and return the transcript
        self._save_to_cache(cache_key, transcript)
        return transcript
    
    def _get_transcript_from_seeking_alpha(self, ticker: str, quarter: str, 
                                         year: int) -> Optional[Dict[str, Any]]:
        """
        Get earnings call transcript from Seeking Alpha.
        
        Args:
            ticker: Company ticker symbol
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year
            
        Returns:
            Dictionary containing transcript data or None if not found
        """
        # This is a simplified implementation
        # In a real-world scenario, you would need to handle authentication,
        # pagination, and more complex parsing
        
        url = f"{self.sources['seeking_alpha']}/{ticker.lower()}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for transcript links matching the quarter and year
            transcript_links = soup.find_all('a', href=re.compile(r'/article/'))
            
            target_pattern = re.compile(f"{ticker}.*{quarter}.*{year}", re.IGNORECASE)
            
            for link in transcript_links:
                if target_pattern.search(link.text):
                    transcript_url = f"https://seekingalpha.com{link['href']}"
                    
                    # Get the transcript page
                    transcript_response = self.session.get(transcript_url)
                    transcript_response.raise_for_status()
                    
                    transcript_soup = BeautifulSoup(transcript_response.text, 'html.parser')
                    
                    # Extract the transcript content
                    transcript_content = transcript_soup.find('div', class_='sa-art')
                    
                    if not transcript_content:
                        continue
                    
                    # Extract different sections
                    sections = {}
                    
                    # Extract opening remarks
                    opening_section = transcript_content.find(text=re.compile('opening remarks|prepared remarks', re.IGNORECASE))
                    if opening_section:
                        opening_div = opening_section.find_parent('div')
                        if opening_div:
                            sections['opening_remarks'] = opening_div.get_text()
                    
                    # Extract Q&A section
                    qa_section = transcript_content.find(text=re.compile('question-and-answer|q&a session', re.IGNORECASE))
                    if qa_section:
                        qa_div = qa_section.find_parent('div')
                        if qa_div:
                            sections['qa_session'] = qa_div.get_text()
                    
                    # Extract full transcript
                    full_transcript = transcript_content.get_text()
                    
                    # Extract participants
                    participants_section = transcript_content.find(text=re.compile('participants|executives|analysts', re.IGNORECASE))
                    participants = []
                    
                    if participants_section:
                        participants_div = participants_section.find_parent('div')
                        if participants_div:
                            participants_text = participants_div.get_text()
                            # Extract names and roles
                            for line in participants_text.split('\n'):
                                if ' - ' in line:
                                    name, role = line.split(' - ', 1)
                                    participants.append({
                                        'name': name.strip(),
                                        'role': role.strip()
                                    })
                    
                    # Extract date
                    date_match = re.search(r'(\w+ \d+, \d{4})', transcript_content.get_text())
                    date = date_match.group(1) if date_match else f"{quarter} {year}"
                    
                    return {
                        "ticker": ticker,
                        "quarter": quarter,
                        "year": year,
                        "date": date,
                        "source": "Seeking Alpha",
                        "url": transcript_url,
                        "participants": participants,
                        "sections": sections,
                        "full_transcript": full_transcript,
                        "found": True
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting transcript from Seeking Alpha: {e}")
            return None
    
    def _get_transcript_from_motley_fool(self, ticker: str, quarter: str, 
                                       year: int) -> Optional[Dict[str, Any]]:
        """
        Get earnings call transcript from Motley Fool.
        
        Args:
            ticker: Company ticker symbol
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year
            
        Returns:
            Dictionary containing transcript data or None if not found
        """
        # Similar implementation to Seeking Alpha but for Motley Fool
        # This is a placeholder - in a real implementation, you would parse the Motley Fool website
        
        logger.info(f"Getting transcript from Motley Fool for {ticker} {quarter} {year}")
        
        # For demonstration purposes, return None to indicate not found
        return None
    
    def extract_speaker_segments(self, transcript: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract segments spoken by different speakers from the transcript.
        
        Args:
            transcript: Transcript dictionary
            
        Returns:
            Dictionary mapping speaker names to lists of their statements
        """
        full_text = transcript.get("full_transcript", "")
        
        # Extract speaker segments using regex
        # This pattern looks for speaker names followed by their statements
        pattern = r'([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Za-z ]+)?)\s*:\s*([^:]+?)(?=(?:[A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Za-z ]+)?)\s*:|$)'
        
        matches = re.findall(pattern, full_text, re.DOTALL)
        
        speaker_segments = {}
        
        for speaker, text in matches:
            speaker = speaker.strip()
            text = text.strip()
            
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            
            speaker_segments[speaker].append(text)
        
        return speaker_segments
    
    def analyze_transcript_sentiment(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment in an earnings call transcript.
        
        Args:
            transcript: Transcript dictionary
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not transcript.get("found", False):
            return {
                "ticker": transcript.get("ticker"),
                "quarter": transcript.get("quarter"),
                "year": transcript.get("year"),
                "error": "Transcript not found"
            }
        
        # Extract speaker segments
        speaker_segments = self.extract_speaker_segments(transcript)
        
        # Analyze sentiment for each speaker
        speaker_sentiment = {}
        
        for speaker, segments in speaker_segments.items():
            # Combine all segments for this speaker
            combined_text = " ".join(segments)
            
            # Analyze sentiment and behavioral patterns
            analysis = self.analyze_text_sentiment(combined_text)
            
            # Store results
            speaker_sentiment[speaker] = analysis
        
        # Analyze sentiment for different sections
        section_sentiment = {}
        
        for section_name, section_text in transcript.get("sections", {}).items():
            section_sentiment[section_name] = self.analyze_text_sentiment(section_text)
        
        # Analyze overall sentiment
        full_transcript = transcript.get("full_transcript", "")
        overall_sentiment = self.analyze_text_sentiment(full_transcript)
        
        # Prepare the result
        result = {
            "ticker": transcript.get("ticker"),
            "quarter": transcript.get("quarter"),
            "year": transcript.get("year"),
            "date": transcript.get("date"),
            "source": transcript.get("source"),
            "url": transcript.get("url"),
            "overall_sentiment": overall_sentiment,
            "speaker_sentiment": speaker_sentiment,
            "section_sentiment": section_sentiment
        }
        
        return result
    
    def analyze_earnings_call_audio(self, audio_file: str) -> Dict[str, Any]:
        """
        Analyze sentiment in earnings call audio.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with sentiment analysis results
        """
        logger.info(f"Analyzing earnings call audio: {audio_file}")
        
        try:
            # Check if file exists
            if not os.path.exists(audio_file):
                return {"error": f"Audio file not found: {audio_file}"}
            
            # Create a cache key
            cache_key = self._get_cache_key("earnings_audio", audio_file)
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached audio analysis for {audio_file}")
                return cached_result
            
            # Convert audio to WAV format if needed
            if not audio_file.endswith('.wav'):
                logger.info(f"Converting audio to WAV format")
                audio = AudioSegment.from_file(audio_file)
                wav_file = audio_file.rsplit('.', 1)[0] + '.wav'
                audio.export(wav_file, format='wav')
                audio_file = wav_file
            
            # Transcribe audio
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                transcript = self.recognizer.recognize_google(audio_data)
            
            # Analyze sentiment
            sentiment_analysis = self.analyze_text_sentiment(transcript)
            
            # Prepare the result
            result = {
                "audio_file": audio_file,
                "transcript": transcript,
                "sentiment_analysis": sentiment_analysis
            }
            
            # Cache and return the result
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"error": str(e)}
    
    def detect_tone_changes(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect changes in tone throughout the earnings call.
        
        Args:
            transcript: Transcript dictionary
            
        Returns:
            Dictionary with tone change analysis results
        """
        if not transcript.get("found", False):
            return {
                "ticker": transcript.get("ticker"),
                "error": "Transcript not found"
            }
        
        # Extract speaker segments
        speaker_segments = self.extract_speaker_segments(transcript)
        
        # Analyze tone changes for key executives
        tone_changes = {}
        
        for speaker, segments in speaker_segments.items():
            # Check if this is a key executive
            is_key_executive = False
            for title in self.key_executives:
                if title.lower() in speaker.lower():
                    is_key_executive = True
                    break
            
            if not is_key_executive and len(segments) < 3:
                continue  # Skip non-executives with few segments
            
            # Analyze each segment separately
            segment_analyses = []
            
            for i, segment in enumerate(segments):
                analysis = self.analyze_text_sentiment(segment)
                
                segment_analyses.append({
                    "segment_index": i,
                    "segment_text": segment[:100] + "..." if len(segment) > 100 else segment,
                    "sentiment": analysis.get("sentiment", {}).get("summary", {}),
                    "behavioral": analysis.get("behavioral", {}).get("summary", {})
                })
            
            # Detect significant changes in sentiment or behavior
            changes = []
            
            for i in range(1, len(segment_analyses)):
                prev = segment_analyses[i-1]
                curr = segment_analyses[i]
                
                # Check for sentiment changes
                prev_sentiment = prev.get("sentiment", {}).get("sentiment", "neutral")
                curr_sentiment = curr.get("sentiment", {}).get("sentiment", "neutral")
                
                prev_score = prev.get("sentiment", {}).get("combined_score", 0)
                curr_score = curr.get("sentiment", {}).get("combined_score", 0)
                
                # Check for behavioral changes
                prev_behavior = prev.get("behavioral", {}).get("behavioral_assessment", "neutral")
                curr_behavior = curr.get("behavioral", {}).get("behavioral_assessment", "neutral")
                
                # Detect significant changes
                if prev_sentiment != curr_sentiment or abs(prev_score - curr_score) > 0.3:
                    changes.append({
                        "type": "sentiment_change",
                        "from_segment": i-1,
                        "to_segment": i,
                        "from_sentiment": prev_sentiment,
                        "to_sentiment": curr_sentiment,
                        "score_change": round(curr_score - prev_score, 2)
                    })
                
                if prev_behavior != curr_behavior:
                    changes.append({
                        "type": "behavioral_change",
                        "from_segment": i-1,
                        "to_segment": i,
                        "from_behavior": prev_behavior,
                        "to_behavior": curr_behavior
                    })
            
            tone_changes[speaker] = {
                "segment_count": len(segments),
                "segment_analyses": segment_analyses,
                "detected_changes": changes
            }
        
        return {
            "ticker": transcript.get("ticker"),
            "quarter": transcript.get("quarter"),
            "year": transcript.get("year"),
            "tone_changes": tone_changes
        }
    
    def extract_guidance_sentiment(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and analyze sentiment specifically in the guidance section.
        
        Args:
            transcript: Transcript dictionary
            
        Returns:
            Dictionary with guidance sentiment analysis
        """
        if not transcript.get("found", False):
            return {
                "ticker": transcript.get("ticker"),
                "error": "Transcript not found"
            }
        
        # Get the guidance section
        guidance_text = transcript.get("sections", {}).get("guidance", "")
        
        # If no explicit guidance section, try to find guidance-related content
        if not guidance_text:
            full_text = transcript.get("full_transcript", "")
            
            # Look for guidance-related paragraphs
            guidance_patterns = [
                r'(?i)(?:guidance|outlook|forecast|projections?|expectations?|anticipate).*?(?:next quarter|next year|future|forward)',
                r'(?i)(?:next quarter|next year|future|forward).*?(?:guidance|outlook|forecast|projections?|expectations?|anticipate)',
                r'(?i)(?:expect|project|forecast|anticipate).*?(?:revenue|earnings|growth|margin|profit)',
                r'(?i)(?:revenue|earnings|growth|margin|profit).*?(?:guidance|outlook|forecast)'
            ]
            
            guidance_paragraphs = []
            
            # Split text into paragraphs
            paragraphs = re.split(r'\n\s*\n', full_text)
            
            for paragraph in paragraphs:
                for pattern in guidance_patterns:
                    if re.search(pattern, paragraph):
                        guidance_paragraphs.append(paragraph)
                        break
            
            guidance_text = "\n\n".join(guidance_paragraphs)
        
        # If still no guidance text, return empty result
        if not guidance_text:
            return {
                "ticker": transcript.get("ticker"),
                "quarter": transcript.get("quarter"),
                "year": transcript.get("year"),
                "guidance_found": False
            }
        
        # Analyze sentiment in guidance text
        guidance_sentiment = self.analyze_text_sentiment(guidance_text)
        
        # Extract specific guidance statements
        guidance_statements = []
        
        # Split into sentences
        sentences = sent_tokenize(guidance_text)
        
        for sentence in sentences:
            # Look for specific guidance patterns
            if re.search(r'(?i)(?:expect|project|forecast|anticipate|guidance|outlook)', sentence):
                guidance_statements.append(sentence)
        
        return {
            "ticker": transcript.get("ticker"),
            "quarter": transcript.get("quarter"),
            "year": transcript.get("year"),
            "guidance_found": True,
            "guidance_text": guidance_text,
            "guidance_sentiment": guidance_sentiment,
            "guidance_statements": guidance_statements
        }
    
    def collect_data(self, ticker: str, quarter: Optional[str] = None, 
                   year: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect and analyze earnings call data for a company.
        
        Args:
            ticker: Company ticker symbol
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year
            
        Returns:
            Dictionary with comprehensive earnings call analysis
        """
        logger.info(f"Collecting earnings call data for {ticker} {quarter} {year}")
        
        # Get the transcript
        transcript = self.get_earnings_call_transcript(ticker, quarter, year)
        
        if not transcript.get("found", False):
            return {
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "error": "Transcript not found"
            }
        
        # Analyze the transcript
        sentiment_analysis = self.analyze_transcript_sentiment(transcript)
        tone_changes = self.detect_tone_changes(transcript)
        guidance_sentiment = self.extract_guidance_sentiment(transcript)
        
        # Combine all analyses
        result = {
            "ticker": ticker,
            "quarter": quarter,
            "year": year,
            "date": transcript.get("date"),
            "source": transcript.get("source"),
            "url": transcript.get("url"),
            "participants": transcript.get("participants"),
            "sentiment_analysis": sentiment_analysis,
            "tone_changes": tone_changes.get("tone_changes", {}),
            "guidance_sentiment": guidance_sentiment
        }
        
        return result
    
    def analyze_data(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze collected earnings call data across multiple calls.
        
        Args:
            data: List of dictionaries containing earnings call data
            
        Returns:
            Dictionary with analysis results
        """
        if not data:
            return {"error": "No data to analyze"}
        
        # Extract company info from first item
        ticker = data[0].get("ticker")
        
        # Prepare result structure
        result = {
            "ticker": ticker,
            "call_count": len(data),
            "time_period": {
                "start": min(item.get("date", "Unknown") for item in data),
                "end": max(item.get("date", "Unknown") for item in data)
            },
            "sentiment_trend": [],
            "guidance_trend": [],
            "executive_behavior_trend": {},
            "summary": {}
        }
        
        # Analyze sentiment trend
        for item in sorted(data, key=lambda x: x.get("date", "")):
            overall_sentiment = item.get("sentiment_analysis", {}).get("overall_sentiment", {})
            sentiment_summary = overall_sentiment.get("sentiment", {}).get("summary", {})
            
            result["sentiment_trend"].append({
                "date": item.get("date"),
                "quarter": item.get("quarter"),
                "year": item.get("year"),
                "sentiment": sentiment_summary.get("sentiment"),
                "score": sentiment_summary.get("combined_score")
            })
        
        # Analyze guidance trend
        for item in sorted(data, key=lambda x: x.get("date", "")):
            guidance = item.get("guidance_sentiment", {})
            
            if guidance.get("guidance_found", False):
                guidance_sentiment = guidance.get("guidance_sentiment", {}).get("sentiment", {}).get("summary", {})
                
                result["guidance_trend"].append({
                    "date": item.get("date"),
                    "quarter": item.get("quarter"),
                    "year": item.get("year"),
                    "sentiment": guidance_sentiment.get("sentiment"),
                    "score": guidance_sentiment.get("combined_score")
                })
        
        # Analyze executive behavior trend
        executives = set()
        
        for item in data:
            for speaker in item.get("tone_changes", {}):
                executives.add(speaker)
        
        for executive in executives:
            result["executive_behavior_trend"][executive] = []
            
            for item in sorted(data, key=lambda x: x.get("date", "")):
                tone_data = item.get("tone_changes", {}).get(executive, {})
                
                if tone_data:
                    # Calculate average sentiment
                    segments = tone_data.get("segment_analyses", [])
                    if segments:
                        avg_sentiment = sum(s.get("sentiment", {}).get("combined_score", 0) for s in segments) / len(segments)
                        
                        # Count behavioral changes
                        behavior_changes = sum(1 for c in tone_data.get("detected_changes", []) if c.get("type") == "behavioral_change")
                        
                        result["executive_behavior_trend"][executive].append({
                            "date": item.get("date"),
                            "quarter": item.get("quarter"),
                            "year": item.get("year"),
                            "avg_sentiment": round(avg_sentiment, 2),
                            "behavior_changes": behavior_changes
                        })
        
        # Generate summary
        if result["sentiment_trend"]:
            # Calculate average sentiment
            avg_sentiment = sum(item.get("score", 0) for item in result["sentiment_trend"]) / len(result["sentiment_trend"])
            
            # Determine sentiment trend
            if len(result["sentiment_trend"]) > 1:
                first = result["sentiment_trend"][0].get("score", 0)
                last = result["sentiment_trend"][-1].get("score", 0)
                trend = last - first
                
                if trend > 0.2:
                    trend_description = "improving"
                elif trend < -0.2:
                    trend_description = "deteriorating"
                else:
                    trend_description = "stable"
            else:
                trend_description = "unknown"
            
            result["summary"]["sentiment"] = {
                "average": round(avg_sentiment, 2),
                "trend": trend_description
            }
        
        return result
