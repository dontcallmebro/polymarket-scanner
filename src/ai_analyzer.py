"""
AI Analyzer
===========
Uses OpenAI GPT-4 to analyze the fundamental aspects of markets.
Provides probability estimates, bias detection, and key factors.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging

from dotenv import load_dotenv

from .database import db

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAnalyzer:
    """
    AI-powered fundamental analysis for Polymarket markets.
    Uses OpenAI GPT-4 to analyze market questions and provide insights.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.db = db
        self.model = "gpt-4o"  # or "gpt-4-turbo" or "gpt-3.5-turbo"
        self._client = None
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment")
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None and self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
        return self._client
    
    def is_available(self) -> bool:
        """Check if AI analysis is available."""
        return self.api_key is not None and self.client is not None
    
    def build_prompt(self, market: Dict) -> str:
        """
        Build analysis prompt for a market.
        """
        question = market.get("question", "Unknown question")
        description = market.get("description", "No description available")
        category = market.get("category", "Unknown")
        end_date = market.get("endDate", market.get("end_date", "Unknown"))
        
        # Parse current prices
        outcome_prices = market.get("outcomePrices", "")
        if isinstance(outcome_prices, str):
            outcome_prices = outcome_prices.replace("[", "").replace("]", "").replace('"', "")
            try:
                prices = [float(p.strip()) for p in outcome_prices.split(",") if p.strip()]
                yes_price = prices[0] if prices else 0.5
            except:
                yes_price = 0.5
        else:
            yes_price = 0.5
        
        prompt = f"""Tu es un analyste expert en marchés prédictifs. Analyse ce pari Polymarket et donne ton évaluation.

MARCHÉ:
Question: {question}
Description: {description[:500] if description else 'Non disponible'}
Catégorie: {category}
Date de résolution: {end_date}
Prix actuel YES: {yes_price:.2f} (le marché donne {yes_price*100:.0f}% de chances)

INSTRUCTIONS:
1. Évalue la probabilité réelle de l'événement selon ton analyse
2. Compare avec le prix du marché pour identifier un biais
3. Identifie les facteurs clés qui influencent ce pari
4. Donne un résumé court et actionable

IMPORTANT: Réponds UNIQUEMENT en JSON valide avec cette structure exacte:
{{
  "ai_probability": 0.XX,
  "confidence": "low|medium|high",
  "bias_direction": "OVERPRICED|UNDERPRICED|FAIR",
  "key_factors": ["facteur1", "facteur2", "facteur3"],
  "summary": "Résumé court de ton analyse (max 100 mots)"
}}

Notes:
- ai_probability: ta probabilité estimée (0.00 à 1.00)
- OVERPRICED: le marché surestime les chances (opportunité de SELL)
- UNDERPRICED: le marché sous-estime les chances (opportunité de BUY)  
- FAIR: le prix reflète correctement la probabilité
"""
        return prompt
    
    def parse_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse AI response JSON.
        """
        try:
            # Find JSON in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            
            if start == -1 or end == 0:
                logger.error("No JSON found in AI response")
                return None
            
            json_str = response_text[start:end]
            result = json.loads(json_str)
            
            # Validate required fields
            required = ["ai_probability", "confidence", "bias_direction", "key_factors", "summary"]
            for field in required:
                if field not in result:
                    logger.error(f"Missing field in AI response: {field}")
                    return None
            
            # Normalize values
            result["ai_probability"] = float(result["ai_probability"])
            result["confidence"] = result["confidence"].lower()
            result["bias_direction"] = result["bias_direction"].upper()
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            return None
    
    def analyze_market(self, market: Dict, force: bool = False) -> Optional[Dict]:
        """
        Analyze a single market using AI.
        
        Args:
            market: Market data dictionary
            force: If True, analyze even if recent analysis exists
        
        Returns:
            Analysis result dictionary or None
        """
        if not self.is_available():
            logger.warning("AI analysis not available (missing API key)")
            return None
        
        market_id = market.get("id")
        
        # Check for recent analysis (within 24h)
        if not force:
            existing = self.db.get_latest_ai_analysis(market_id)
            if existing:
                analyzed_at = datetime.fromisoformat(existing["analyzed_at"])
                if datetime.utcnow() - analyzed_at < timedelta(hours=24):
                    logger.debug(f"Using cached AI analysis for {market_id}")
                    return existing
        
        # Build prompt
        prompt = self.build_prompt(market)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un analyste expert en marchés prédictifs et paris. Réponds toujours en JSON valide."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            result = self.parse_response(response_text)
            
            if result:
                result["model_used"] = self.model
                
                # Store in database
                self.db.insert_ai_analysis(market_id, result)
                
                logger.info(f"AI analysis complete for {market_id}: {result['bias_direction']}")
                return result
            
        except Exception as e:
            logger.error(f"AI analysis failed for {market_id}: {e}")
        
        return None
    
    def batch_analyze(self, markets: List[Dict], limit: int = 10) -> List[Dict]:
        """
        Analyze multiple markets (with limit to control API costs).
        
        Args:
            markets: List of market dictionaries
            limit: Maximum number of markets to analyze
        
        Returns:
            List of analysis results
        """
        if not self.is_available():
            logger.warning("AI analysis not available")
            return []
        
        results = []
        count = 0
        
        for market in markets:
            if count >= limit:
                break
            
            result = self.analyze_market(market)
            if result:
                result["market_id"] = market.get("id")
                result["question"] = market.get("question", "")[:100]
                results.append(result)
                count += 1
        
        logger.info(f"Batch AI analysis complete: {len(results)}/{limit} markets")
        return results
    
    def get_ai_insights_summary(self, analyses: List[Dict]) -> Dict:
        """
        Summarize AI insights across multiple markets.
        """
        if not analyses:
            return {}
        
        overpriced = [a for a in analyses if a.get("bias_direction") == "OVERPRICED"]
        underpriced = [a for a in analyses if a.get("bias_direction") == "UNDERPRICED"]
        fair = [a for a in analyses if a.get("bias_direction") == "FAIR"]
        
        high_conf = [a for a in analyses if a.get("confidence") == "high"]
        
        return {
            "total_analyzed": len(analyses),
            "overpriced_count": len(overpriced),
            "underpriced_count": len(underpriced),
            "fair_count": len(fair),
            "high_confidence_count": len(high_conf),
            "best_buy_opportunities": [
                {"question": a["question"], "ai_prob": a["ai_probability"]}
                for a in underpriced if a.get("confidence") in ["medium", "high"]
            ][:3],
            "best_sell_opportunities": [
                {"question": a["question"], "ai_prob": a["ai_probability"]}
                for a in overpriced if a.get("confidence") in ["medium", "high"]
            ][:3]
        }


# Singleton instance
ai_analyzer = AIAnalyzer()


if __name__ == "__main__":
    print("Testing AI Analyzer...")
    print(f"AI Available: {ai_analyzer.is_available()}")
    
    if ai_analyzer.is_available():
        # Test with a sample market
        from .api_client import client
        
        markets = client.get_markets(limit=1)
        if markets:
            market = markets[0]
            print(f"\nAnalyzing: {market.get('question', 'N/A')[:60]}...")
            
            result = ai_analyzer.analyze_market(market)
            if result:
                print(f"AI Probability: {result['ai_probability']:.0%}")
                print(f"Bias: {result['bias_direction']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Summary: {result['summary']}")
    else:
        print("Set OPENAI_API_KEY in config/.env to enable AI analysis")
