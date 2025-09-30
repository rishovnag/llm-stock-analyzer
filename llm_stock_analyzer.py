# Complete LLM Stock Analysis System with Movement Thresholds
# File: llm_stock_analyzer.py

import pandas as pd
import numpy as np
import json
import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Structure for prediction results"""
    model_name: str
    prediction_type: str  # 'direction', 'sig_move_1pct', 'sig_move_2pct', 'sig_move_3pct'
    prediction: int  # 1 for UP/SIGNIFICANT, 0 for DOWN/NOT_SIGNIFICANT
    confidence: float
    reasoning: str
    timestamp: str
    technical_data: Dict[str, float]
    movement_threshold: float

class MovementAnalyzer:
    """Analyzes different types of movements and thresholds"""
    
    @staticmethod
    def get_available_datasets(stock_symbol: str, data_dir: Path) -> Dict[str, Path]:
        """Find all available datasets for a stock symbol"""
        datasets = {}
        
        # Movement types to look for
        movement_types = {
            'direction': ['Direction'],
            'sig_move_1pct': ['SigMove_1pct', 'SigMove_neg1pct'],
            'sig_move_2pct': ['SigMove_2pct', 'SigMove_neg2pct'],
            'sig_move_3pct': ['SigMove_3pct', 'SigMove_neg3pct']
        }
        
        # Find matching files
        for file_path in data_dir.glob(f"{stock_symbol}*.csv"):
            filename = file_path.name
            
            for movement_type, patterns in movement_types.items():
                for pattern in patterns:
                    if pattern in filename:
                        if movement_type not in datasets:
                            datasets[movement_type] = file_path
                        break
        
        logger.info(f"Available datasets for {stock_symbol}: {list(datasets.keys())}")
        return datasets
    
    @staticmethod
    def calculate_movement_stats(df: pd.DataFrame, movement_column: str) -> Dict[str, float]:
        """Calculate statistics for movement predictions"""
        if movement_column not in df.columns:
            return {}
        
        movements = df[movement_column].dropna()
        if len(movements) == 0:
            return {}
        
        stats = {
            'total_days': len(movements),
            'positive_days': (movements == 1).sum(),
            'negative_days': (movements == 0).sum(),
            'positive_rate': (movements == 1).mean(),
            'recent_10_positive_rate': (movements.tail(10) == 1).mean() if len(movements) >= 10 else 0,
            'recent_30_positive_rate': (movements.tail(30) == 1).mean() if len(movements) >= 30 else 0,
            'consecutive_positive': 0,
            'consecutive_negative': 0
        }
        
        # Calculate consecutive streaks
        if len(movements) > 0:
            current_streak = 0
            for i in reversed(range(len(movements))):
                if movements.iloc[i] == movements.iloc[-1]:
                    current_streak += 1
                else:
                    break
            
            if movements.iloc[-1] == 1:
                stats['consecutive_positive'] = current_streak
            else:
                stats['consecutive_negative'] = current_streak
        
        return stats

class TechnicalAnalyzer:
    """Enhanced Technical Analysis Calculator"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive technical indicators from price data"""
        if len(df) < 20:
            return {}
        
        try:
            close = df['Close'].values
            high = df['High'].values if 'High' in df.columns else close
            low = df['Low'].values if 'Low' in df.columns else close
            volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(close))
            
            # Convert volume to numeric if it's string
            if len(volume) > 0 and isinstance(volume[0], (str, np.str_)):
                volume_series = pd.Series(volume.astype(str))
                volume = pd.to_numeric(volume_series.str.replace(',', ''), errors='coerce').fillna(0).values
            
            indicators = {}
            
            # Basic price information
            indicators['current_price'] = float(close[-1])
            indicators['prev_close'] = float(close[-2]) if len(close) > 1 else float(close[-1])
            indicators['daily_change'] = ((close[-1] - close[-2]) / close[-2] * 100) if len(close) > 1 else 0.0
            indicators['daily_high'] = float(high[-1]) if len(high) > 0 else indicators['current_price']
            indicators['daily_low'] = float(low[-1]) if len(low) > 0 else indicators['current_price']
            indicators['daily_range'] = ((indicators['daily_high'] - indicators['daily_low']) / indicators['current_price'] * 100)
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                if len(close) >= period:
                    sma = float(np.mean(close[-period:]))
                    indicators[f'sma_{period}'] = sma
                    indicators[f'price_vs_sma{period}'] = ((close[-1] - sma) / sma * 100)
            
            # Exponential moving averages
            if len(close) >= 12:
                ema_12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
                indicators['ema_12'] = float(ema_12)
                indicators['price_vs_ema12'] = ((close[-1] - ema_12) / ema_12 * 100)
            
            if len(close) >= 26:
                ema_26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
                indicators['ema_26'] = float(ema_26)
                # MACD
                if 'ema_12' in indicators:
                    indicators['macd'] = indicators['ema_12'] - ema_26
            
            # Volatility measures
            for period in [5, 10, 20]:
                if len(close) >= period + 1:
                    returns = np.diff(close[-period-1:]) / close[-period-1:-1]
                    indicators[f'volatility_{period}d'] = float(np.std(returns) * np.sqrt(252) * 100)
                    indicators[f'avg_true_range_{period}d'] = float(np.mean(np.abs(returns)) * 100)
            
            # RSI calculation
            if len(close) >= 15:
                deltas = np.diff(close[-15:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    indicators['rsi'] = float(100 - (100 / (1 + rs)))
                else:
                    indicators['rsi'] = 100.0
            
            # Support and Resistance levels
            if len(close) >= 20:
                recent_data = close[-20:]
                indicators['support_level'] = float(np.min(recent_data))
                indicators['resistance_level'] = float(np.max(recent_data))
                indicators['distance_from_support'] = ((close[-1] - indicators['support_level']) / indicators['support_level'] * 100)
                indicators['distance_from_resistance'] = ((indicators['resistance_level'] - close[-1]) / close[-1] * 100)
            
            # Momentum indicators
            for period in [3, 5, 10]:
                if len(close) >= period + 1:
                    indicators[f'momentum_{period}d'] = ((close[-1] - close[-period-1]) / close[-period-1] * 100)
                    indicators[f'rate_of_change_{period}d'] = indicators[f'momentum_{period}d']
            
            # Bollinger Bands
            if len(close) >= 20:
                bb_period = 20
                bb_std = 2
                sma_bb = np.mean(close[-bb_period:])
                std_bb = np.std(close[-bb_period:])
                
                indicators['bb_upper'] = sma_bb + (bb_std * std_bb)
                indicators['bb_lower'] = sma_bb - (bb_std * std_bb)
                indicators['bb_middle'] = sma_bb
                indicators['bb_position'] = ((close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']) * 100)
            
            # Volume indicators
            if len(volume) >= 10 and volume.sum() > 0:
                indicators['current_volume'] = float(volume[-1])
                indicators['avg_volume_10d'] = float(np.mean(volume[-10:]))
                indicators['avg_volume_20d'] = float(np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume))
                indicators['volume_ratio'] = float(volume[-1] / indicators['avg_volume_10d']) if indicators['avg_volume_10d'] > 0 else 1.0
                
                # Volume momentum
                if len(volume) >= 5:
                    indicators['volume_momentum_5d'] = float((np.mean(volume[-5:]) - np.mean(volume[-10:-5])) / np.mean(volume[-10:-5]) * 100) if len(volume) >= 10 else 0
            
            # Price patterns and levels
            if len(high) >= 5 and len(low) >= 5:
                # Recent highs and lows
                indicators['highest_5d'] = float(np.max(high[-5:]))
                indicators['lowest_5d'] = float(np.min(low[-5:]))
                indicators['distance_from_5d_high'] = ((indicators['highest_5d'] - close[-1]) / close[-1] * 100)
                indicators['distance_from_5d_low'] = ((close[-1] - indicators['lowest_5d']) / indicators['lowest_5d'] * 100)
            
            # Trend strength
            if len(close) >= 10:
                # Linear regression slope
                x = np.arange(10)
                y = close[-10:]
                slope, _ = np.polyfit(x, y, 1)
                indicators['trend_strength'] = float(slope / close[-1] * 100 * 10)  # Normalized slope
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}

class UnifiedLLMInterface:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self, provider: str, api_key: str, model_config: Dict):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_config = model_config
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        try:
            if self.provider == 'openai':
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                
            elif self.provider == 'anthropic':
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                
            elif self.provider == 'google':
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_config.get('model', 'gemini-1.5-flash'))
                
            logger.info(f"Successfully initialized {self.provider} client")
            
        except ImportError as e:
            logger.error(f"Failed to import {self.provider} library: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            raise
    
    async def predict(self, prompt: str) -> Dict[str, Any]:
        """Make prediction using the specified LLM"""
        try:
            if self.provider == 'openai':
                return await self._predict_openai(prompt)
            elif self.provider == 'anthropic':
                return await self._predict_anthropic(prompt)
            elif self.provider == 'google':
                return await self._predict_google(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Prediction failed for {self.provider}: {e}")
            return {
                "prediction": 0,
                "confidence": 0.5,
                "reasoning": f"Error: {str(e)}",
                "error": True
            }
    
    async def _predict_openai(self, prompt: str) -> Dict[str, Any]:
        """OpenAI prediction"""
        response = self.client.chat.completions.create(
            model=self.model_config.get('model', 'gpt-4o-mini'),
            messages=[
                {"role": "system", "content": "You are a financial analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.model_config.get('temperature', 0.3),
            max_tokens=self.model_config.get('max_tokens', 500)
        )
        
        response_text = response.choices[0].message.content
        return self._parse_json_response(response_text)
    
    async def _predict_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Anthropic Claude prediction"""
        message = self.client.messages.create(
            model=self.model_config.get('model', 'claude-3-haiku-20240307'),
            max_tokens=self.model_config.get('max_tokens', 500),
            temperature=self.model_config.get('temperature', 0.3),
            messages=[
                {"role": "user", "content": f"You are a financial analyst. Always respond with valid JSON only.\n\n{prompt}"}
            ]
        )
        
        response_text = message.content[0].text
        return self._parse_json_response(response_text)
    
    async def _predict_google(self, prompt: str) -> Dict[str, Any]:
        """Google Gemini prediction"""
        full_prompt = f"You are a financial analyst. Always respond with valid JSON only.\n\n{prompt}"
        
        generation_config = {
            'temperature': self.model_config.get('temperature', 0.3),
            'max_output_tokens': self.model_config.get('max_tokens', 500),
        }
        
        response = self.client.generate_content(full_prompt, generation_config=generation_config)
        return self._parse_json_response(response.text)
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[start_idx:end_idx + 1]
            parsed = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['prediction', 'confidence']
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure prediction is 0 or 1
            parsed['prediction'] = 1 if float(parsed['prediction']) > 0.5 else 0
            
            # Ensure confidence is between 0 and 1
            parsed['confidence'] = max(0.0, min(1.0, float(parsed['confidence'])))
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {
                "prediction": 0,
                "confidence": 0.5,
                "reasoning": "Failed to parse response",
                "error": True
            }

class LLMStockPredictor:
    """Main LLM Stock Prediction System with Movement Thresholds"""
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = Path(data_directory)
        self.technical_analyzer = TechnicalAnalyzer()
        self.movement_analyzer = MovementAnalyzer()
        self.llm_interfaces = {}
        self.load_api_keys()
        self.setup_models()
    
    def load_api_keys(self):
        """Load API keys from environment variables"""
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_keys = {
            'openai': os.getenv('OPENAI_API_KEY', ''),
            'anthropic': os.getenv('ANTHROPIC_API_KEY', ''),
            'google': os.getenv('GOOGLE_API_KEY', ''),
        }
        
        # Check which keys are available
        available_keys = {k: v for k, v in self.api_keys.items() if v}
        logger.info(f"Available API keys: {list(available_keys.keys())}")
    
    def setup_models(self):
        """Setup LLM model configurations"""
        self.model_configs = {
            'gpt4o-mini': {
                'provider': 'openai',
                'model': 'gpt-4o-mini',
                'temperature': 0.3,
                'max_tokens': 500
            },
            'gpt4o': {
                'provider': 'openai',
                'model': 'gpt-4o',
                'temperature': 0.3,
                'max_tokens': 500
            },
            'claude-haiku': {
                'provider': 'anthropic',
                'model': 'claude-3-haiku-20240307',
                'temperature': 0.3,
                'max_tokens': 500
            },
            'claude-sonnet': {
                'provider': 'anthropic',
                'model': 'claude-3-5-sonnet-20241022',
                'temperature': 0.3,
                'max_tokens': 500
            },
            'gemini-flash': {
                'provider': 'google',
                'model': 'gemini-1.5-flash',
                'temperature': 0.3,
                'max_tokens': 500
            },
            'gemini-pro': {
                'provider': 'google',
                'model': 'gemini-1.5-pro',
                'temperature': 0.3,
                'max_tokens': 500
            },
        }
    
    def initialize_llm(self, model_name: str) -> UnifiedLLMInterface:
        """Initialize specific LLM interface"""
        if model_name in self.llm_interfaces:
            return self.llm_interfaces[model_name]
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        provider = config['provider']
        
        if not self.api_keys.get(provider):
            raise ValueError(f"API key not found for {provider}")
        
        interface = UnifiedLLMInterface(provider, self.api_keys[provider], config)
        self.llm_interfaces[model_name] = interface
        
        return interface
    
    def get_movement_datasets(self, stock_symbol: str) -> Dict[str, Tuple[pd.DataFrame, str]]:
        """Load all available movement datasets for a stock"""
        datasets = self.movement_analyzer.get_available_datasets(stock_symbol, self.data_directory)
        loaded_datasets = {}
        
        for movement_type, file_path in datasets.items():
            try:
                df = pd.read_csv(file_path)
                
                # Convert Date column
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                
                # Convert price columns to numeric
                price_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
                for col in price_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Determine movement column
                movement_column = None
                if 'Direction' in df.columns:
                    movement_column = 'Direction'
                elif 'Significant_Move' in df.columns:
                    movement_column = 'Significant_Move'
                
                if movement_column:
                    loaded_datasets[movement_type] = (df, movement_column)
                    logger.info(f"Loaded {movement_type} dataset: {len(df)} rows, column: {movement_column}")
                
            except Exception as e:
                logger.error(f"Error loading {movement_type} dataset: {e}")
        
        return loaded_datasets
    
    def create_movement_prediction_prompt(self, stock_symbol: str, movement_type: str, 
                                        technical_data: Dict[str, float], 
                                        movement_stats: Dict[str, float]) -> str:
        """Create specific prompt for movement type prediction"""
        
        # Define movement type descriptions and thresholds
        movement_descriptions = {
            'direction': {
                'description': 'basic directional movement (UP or DOWN)',
                'threshold': 0.0,
                'task': 'Predict if the index will close UP (higher) or DOWN (lower) than previous close'
            },
            'sig_move_1pct': {
                'description': 'significant movement greater than 1%',
                'threshold': 1.0,
                'task': 'Predict if the index will have a SIGNIFICANT move (>1% up or down) or stay within 1% range'
            },
            'sig_move_2pct': {
                'description': 'significant movement greater than 2%',
                'threshold': 2.0,
                'task': 'Predict if the index will have a SIGNIFICANT move (>2% up or down) or stay within 2% range'
            },
            'sig_move_3pct': {
                'description': 'significant movement greater than 3%',
                'threshold': 3.0,
                'task': 'Predict if the index will have a SIGNIFICANT move (>3% up or down) or stay within 3% range'
            }
        }
        
        movement_info = movement_descriptions.get(movement_type, movement_descriptions['direction'])
        
        prompt = f"""
Analyze {stock_symbol} index for {movement_info['description']} prediction.

CURRENT TECHNICAL DATA:
- Current Price: ₹{technical_data.get('current_price', 0):.2f}
- Daily Change: {technical_data.get('daily_change', 0):.2f}%
- Daily Range: {technical_data.get('daily_range', 0):.2f}%
- Volatility (10-day): {technical_data.get('volatility_10d', 0):.2f}%

MOMENTUM INDICATORS:
- RSI (14): {technical_data.get('rsi', 50):.1f}
- 3-day Momentum: {technical_data.get('momentum_3d', 0):.2f}%
- 5-day Momentum: {technical_data.get('momentum_5d', 0):.2f}%
- 10-day Momentum: {technical_data.get('momentum_10d', 0):.2f}%
- Trend Strength: {technical_data.get('trend_strength', 0):.2f}%

MOVING AVERAGES:
- Price vs SMA(5): {technical_data.get('price_vs_sma5', 0):.2f}%
- Price vs SMA(10): {technical_data.get('price_vs_sma10', 0):.2f}%
- Price vs SMA(20): {technical_data.get('price_vs_sma20', 0):.2f}%
- Price vs EMA(12): {technical_data.get('price_vs_ema12', 0):.2f}%

SUPPORT/RESISTANCE:
- Distance from Support: {technical_data.get('distance_from_support', 0):.2f}%
- Distance from Resistance: {technical_data.get('distance_from_resistance', 0):.2f}%
- Distance from 5-day High: {technical_data.get('distance_from_5d_high', 0):.2f}%
- Distance from 5-day Low: {technical_data.get('distance_from_5d_low', 0):.2f}%

BOLLINGER BANDS:
- BB Position: {technical_data.get('bb_position', 50):.1f}%

VOLUME ANALYSIS:
- Volume Ratio: {technical_data.get('volume_ratio', 1):.2f}x
- Volume Momentum (5d): {technical_data.get('volume_momentum_5d', 0):.2f}%

HISTORICAL MOVEMENT STATISTICS:
- Total Trading Days: {movement_stats.get('total_days', 0)}
- Positive Movement Rate: {movement_stats.get('positive_rate', 0.5):.1%}
- Recent 10-day Rate: {movement_stats.get('recent_10_positive_rate', 0.5):.1%}
- Recent 30-day Rate: {movement_stats.get('recent_30_positive_rate', 0.5):.1%}
- Current Streak (Positive): {movement_stats.get('consecutive_positive', 0)} days
- Current Streak (Negative): {movement_stats.get('consecutive_negative', 0)} days

PREDICTION TASK:
{movement_info['task']}

For movements >{movement_info['threshold']}%, consider:
- Current volatility level vs historical
- Momentum strength and direction
- Support/resistance proximity
- Volume patterns
- Recent streak patterns
- Market structure (trending vs ranging)

Output format (ONLY valid JSON):
{{
  "prediction": 1,  // 1 for UP/SIGNIFICANT, 0 for DOWN/NOT_SIGNIFICANT
  "confidence": 0.75,  // 0.0 to 1.0
  "reasoning": "Brief explanation focusing on {movement_info['description']}"
}}
"""
        return prompt
    
    async def predict_single_movement(self, model_name: str, stock_symbol: str, 
                                    movement_type: str, df: pd.DataFrame, 
                                    movement_column: str) -> PredictionResult:
        """Make prediction for a specific movement type"""
        try:
            # Calculate technical indicators
            technical_data = self.technical_analyzer.calculate_indicators(df)
            
            # Calculate movement statistics
            movement_stats = self.movement_analyzer.calculate_movement_stats(df, movement_column)
            
            # Initialize LLM
            llm_interface = self.initialize_llm(model_name)
            
            # Create movement-specific prompt
            prompt = self.create_movement_prediction_prompt(
                stock_symbol, movement_type, technical_data, movement_stats
            )
            
            # Get prediction
            logger.info(f"Getting {movement_type} prediction from {model_name}...")
            result = await llm_interface.predict(prompt)
            
            # Determine threshold
            threshold_map = {
                'direction': 0.0,
                'sig_move_1pct': 1.0,
                'sig_move_2pct': 2.0,
                'sig_move_3pct': 3.0
            }
            
            # Create result object
            prediction_result = PredictionResult(
                model_name=model_name,
                prediction_type=movement_type,
                prediction=result.get('prediction', 0),
                confidence=result.get('confidence', 0.5),
                reasoning=result.get('reasoning', 'No reasoning provided'),
                timestamp=datetime.now().isoformat(),
                technical_data=technical_data,
                movement_threshold=threshold_map.get(movement_type, 0.0)
            )
            
            logger.info(f"{model_name} {movement_type} prediction: "
                       f"{'POSITIVE' if prediction_result.prediction == 1 else 'NEGATIVE'} "
                       f"(confidence: {prediction_result.confidence:.2f})")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error getting {movement_type} prediction from {model_name}: {e}")
            return PredictionResult(
                model_name=model_name,
                prediction_type=movement_type,
                prediction=0,
                confidence=0.5,
                reasoning=f"Error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                technical_data={},
                movement_threshold=0.0
            )
    
    async def predict_all_movements(self, model_names: List[str], stock_symbol: str) -> Dict[str, List[PredictionResult]]:
        """Get predictions for all available movement types"""
        # Load all available datasets
        datasets = self.get_movement_datasets(stock_symbol)
        
        if not datasets:
            raise FileNotFoundError(f"No datasets found for {stock_symbol}")
        
        results = {}
        
        # Process each movement type
        for movement_type, (df, movement_column) in datasets.items():
            logger.info(f"\n📊 Processing {movement_type} predictions for {stock_symbol}...")
            
            # Get predictions from all models for this movement type
            tasks = [
                self.predict_single_movement(model_name, stock_symbol, movement_type, df, movement_column)
                for model_name in model_names
            ]
            
            movement_results = await asyncio.gather(*tasks)
            results[movement_type] = movement_results
        
        return results
    
    def display_movement_results(self, stock_symbol: str, all_results: Dict[str, List[PredictionResult]]):
        """Display results organized by movement type"""
        print("\n" + "="*100)
        print(f"🤖 LLM STOCK PREDICTIONS FOR {stock_symbol}")
        print("="*100)
        
        movement_names = {
            'direction': 'DIRECTIONAL MOVEMENT (Up/Down)',
            'sig_move_1pct': 'SIGNIFICANT MOVEMENT >1%',
            'sig_move_2pct': 'SIGNIFICANT MOVEMENT >2%',
            'sig_move_3pct': 'SIGNIFICANT MOVEMENT >3%'
        }
        
        for movement_type, results in all_results.items():
            print(f"\n📈 {movement_names.get(movement_type, movement_type.upper())}")
            print("-" * 80)
            
            positive_predictions = 0
            total_confidence_pos = 0
            total_confidence_neg = 0
            
            for result in results:
                if movement_type == 'direction':
                    direction = "📈 UP" if result.prediction == 1 else "📉 DOWN"
                else:
                    direction = "🚀 SIGNIFICANT" if result.prediction == 1 else "📊 NORMAL"
                
                confidence_bar = "█" * int(result.confidence * 20)
                
                print(f"{result.model_name:15s} | {direction:13s} | {result.confidence:5.1%} | {confidence_bar}")
                print(f"                  {result.reasoning}")
                
                if result.prediction == 1:
                    positive_predictions += 1
                    total_confidence_pos += result.confidence
                else:
                    total_confidence_neg += result.confidence
            
            # Consensus for this movement type
            negative_predictions = len(results) - positive_predictions
            
            print(f"\n🎯 CONSENSUS:")
            if positive_predictions > negative_predictions:
                avg_confidence = total_confidence_pos / positive_predictions if positive_predictions > 0 else 0
                if movement_type == 'direction':
                    print(f"   BULLISH: {positive_predictions}/{len(results)} models predict UP (avg: {avg_confidence:.1%})")
                else:
                    print(f"   HIGH VOLATILITY: {positive_predictions}/{len(results)} models predict SIGNIFICANT move (avg: {avg_confidence:.1%})")
            elif negative_predictions > positive_predictions:
                avg_confidence = total_confidence_neg / negative_predictions if negative_predictions > 0 else 0
                if movement_type == 'direction':
                    print(f"   BEARISH: {negative_predictions}/{len(results)} models predict DOWN (avg: {avg_confidence:.1%})")
                else:
                    print(f"   LOW VOLATILITY: {negative_predictions}/{len(results)} models predict NORMAL range (avg: {avg_confidence:.1%})")
            else:
                print(f"   MIXED SIGNALS: {positive_predictions} vs {negative_predictions}")
        
        print("\n" + "="*100)
    
    def save_movement_results(self, stock_symbol: str, all_results: Dict[str, List[PredictionResult]]):
        """Save all movement results to JSON file"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f"{stock_symbol}_all_movements_{timestamp}.json"
        
        # Convert results to dict for JSON serialization
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "stock_symbol": stock_symbol,
            "movement_predictions": {}
        }
        
        for movement_type, results in all_results.items():
            results_dict["movement_predictions"][movement_type] = [
                {
                    "model_name": r.model_name,
                    "prediction_type": r.prediction_type,
                    "prediction": r.prediction,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "timestamp": r.timestamp,
                    "movement_threshold": r.movement_threshold,
                    "technical_data": r.technical_data
                }
                for r in results
            ]
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"All movement results saved to: {filename}")

# Example usage functions
async def test_all_movements():
    """Test all movement types for a stock"""
    predictor = LLMStockPredictor(".")
    
    # Test with available models (will skip if API key not available)
    models_to_test = ['gemini-flash']  # Start with free model
    
    try:
        # Test NIFTY50 with all available movement types
        all_results = await predictor.predict_all_movements(models_to_test, 'NIFTY50')
        predictor.display_movement_results('NIFTY50', all_results)
        predictor.save_movement_results('NIFTY50', all_results)
        
    except Exception as e:
        print(f"Error: {e}")

async def comprehensive_analysis():
    """Comprehensive analysis across multiple stocks and models"""
    predictor = LLMStockPredictor(".")
    
    # Models to test (add more as you get API keys)
    models = ['gemini-flash', 'gpt4o-mini', 'claude-haiku']
    
    # Stocks to analyze
    stocks = ['NIFTY50', 'NIFTYBANK', 'NIFTYIT']
    
    for stock in stocks:
        try:
            print(f"\n🔍 COMPREHENSIVE ANALYSIS FOR {stock}")
            print("="*60)
            
            all_results = await predictor.predict_all_movements(models, stock)
            predictor.display_movement_results(stock, all_results)
            predictor.save_movement_results(stock, all_results)
            
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            continue

if __name__ == "__main__":
    print("🚀 LLM Stock Movement Prediction System")
    print("Supports: Direction, 1%, 2%, and 3% movement thresholds")
    print("\nChoose an option:")
    print("1. Test all movement types (single model)")
    print("2. Comprehensive analysis (multiple models & stocks)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_all_movements())
    elif choice == "2":
        asyncio.run(comprehensive_analysis())
    else:
        print("Invalid choice. Running test...")
        asyncio.run(test_all_movements())