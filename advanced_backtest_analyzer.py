# Advanced Backtesting System with Variable Train-Test Splits and Rolling Windows
# File: advanced_backtest_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
import itertools
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestRun:
    """Structure for individual backtest run results"""
    run_id: int
    stock_symbol: str
    movement_type: str
    split_type: str  # 'fixed' or 'rolling'
    train_ratio: Optional[float]  # For fixed splits
    train_size: int
    test_size: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    volatility: float
    total_predictions: int

class AdvancedBacktestEngine:
    """Advanced backtesting engine with multiple split strategies"""
    
    def __init__(self, data_directory: str = "data",
                 prediction_mode: str = "ensemble",
                 ensemble_models=None,
                 tau: float = 0.5,
                 use_tiered_weights: bool = False,
                 cache_dir: str = "llm_cache"):
        """
        prediction_mode:
            "ensemble" (default) -> next-day predictions come from the
                confidence-weighted LLM ensemble (Eq. 5), exactly as described
                in the paper. This is what generates the reported tables.
            "rule"               -> legacy hand-coded technical-rule predictor,
                retained only as a cheap ablation / smoke-test baseline. It does
                NOT reproduce the paper's headline numbers.
        """
        self.data_directory = Path(data_directory)
        self.prediction_mode = prediction_mode

        # Import required modules
        try:
            from llm_stock_analyzer import LLMStockPredictor, TechnicalAnalyzer, MovementAnalyzer
            self.technical_analyzer = TechnicalAnalyzer()
            self.movement_analyzer = MovementAnalyzer()
            self.predictor = LLMStockPredictor(data_directory)
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise

        # The LLM ensemble (Eq. 5). Instantiated lazily-but-eagerly here so the
        # engine fails fast if the ensemble cannot be constructed.
        self.ensemble = None
        if self.prediction_mode == "ensemble":
            from ensemble_predictor import EnsemblePredictor
            self.ensemble = EnsemblePredictor(
                data_directory=str(data_directory),
                models=ensemble_models,
                tau=tau,
                use_tiered_weights=use_tiered_weights,
                cache_dir=cache_dir,
            )
            logger.info(f"Backtest prediction mode: ENSEMBLE "
                        f"({len(self.ensemble.models)} models, tau={tau}, "
                        f"tiered_weights={use_tiered_weights})")
        else:
            logger.info("Backtest prediction mode: RULE (legacy ablation baseline)")

    def _predict(self, stock_symbol: str, movement_type: str,
                 technical_data: dict, movement_stats: dict):
        """Single dispatch point for next-day prediction used by every split.

        In ensemble mode this calls the confidence-weighted LLM ensemble; in
        rule mode it falls back to the legacy technical-rule predictor.
        """
        if self.prediction_mode == "ensemble":
            return self.ensemble.predict_day(
                stock_symbol, movement_type, technical_data, movement_stats
            )
        return self.technical_rule_prediction(
            technical_data, movement_stats, movement_type
        )
    
    def technical_rule_prediction(self, technical_data: Dict[str, float], 
                                movement_stats: Dict[str, float], 
                                movement_type: str) -> Tuple[int, float]:
        """Generate rule-based predictions from technical indicators"""
        
        if not technical_data:
            return 0, 0.5
        
        # Get key indicators with defaults
        rsi = technical_data.get('rsi', 50)
        daily_change = technical_data.get('daily_change', 0)
        momentum_3d = technical_data.get('momentum_3d', 0)
        momentum_5d = technical_data.get('momentum_5d', 0)
        momentum_10d = technical_data.get('momentum_10d', 0)
        price_vs_sma5 = technical_data.get('price_vs_sma5', 0)
        price_vs_sma10 = technical_data.get('price_vs_sma10', 0)
        price_vs_sma20 = technical_data.get('price_vs_sma20', 0)
        volatility_10d = technical_data.get('volatility_10d', 10)
        volume_ratio = technical_data.get('volume_ratio', 1)
        bb_position = technical_data.get('bb_position', 50)
        trend_strength = technical_data.get('trend_strength', 0)
        
        # Historical context
        positive_rate = movement_stats.get('positive_rate', 0.5)
        recent_10_rate = movement_stats.get('recent_10_positive_rate', 0.5)
        
        if movement_type == 'direction':
            # Predict UP (1) or DOWN (0)
            signals = []
            
            # RSI signals
            if rsi > 70:
                signals.append(-0.3)  # Overbought, expect reversal
            elif rsi > 50:
                signals.append(0.2)   # Bullish momentum
            elif rsi < 30:
                signals.append(0.3)   # Oversold, expect bounce
            else:
                signals.append(-0.1)  # Bearish momentum
            
            # Momentum signals
            momentum_score = (momentum_3d + momentum_5d + momentum_10d) / 3
            if momentum_score > 1:
                signals.append(0.4)
            elif momentum_score > 0:
                signals.append(0.2)
            elif momentum_score < -1:
                signals.append(-0.4)
            else:
                signals.append(-0.2)
            
            # Trend signals
            trend_score = (price_vs_sma5 + price_vs_sma10 + price_vs_sma20) / 3
            if trend_score > 2:
                signals.append(0.3)
            elif trend_score > 0:
                signals.append(0.15)
            elif trend_score < -2:
                signals.append(-0.3)
            else:
                signals.append(-0.15)
            
            # Volume confirmation
            if volume_ratio > 1.5:
                signals.append(0.2)
            elif volume_ratio > 1.2:
                signals.append(0.1)
            elif volume_ratio < 0.8:
                signals.append(-0.1)
            
            # Bollinger Bands position
            if bb_position > 80:
                signals.append(-0.2)  # Near upper band
            elif bb_position < 20:
                signals.append(0.2)   # Near lower band
            
            # Recent performance vs historical
            if recent_10_rate > positive_rate + 0.1:
                signals.append(0.1)
            elif recent_10_rate < positive_rate - 0.1:
                signals.append(-0.1)
            
            # Combine signals
            total_signal = sum(signals)
            prediction = 1 if total_signal > 0 else 0
            confidence = min(0.95, max(0.55, 0.6 + abs(total_signal) * 0.2))
            
        else:
            # Predict SIGNIFICANT (1) or NORMAL (0) movement
            threshold_map = {
                'sig_move_1pct': 1.0,
                'sig_move_2pct': 2.0,
                'sig_move_3pct': 3.0
            }
            threshold = threshold_map.get(movement_type, 1.0)
            
            # Volatility is key for significant moves
            volatility_signal = min(1.0, volatility_10d / (15 * threshold))
            
            # Strong momentum increases significant move probability
            abs_momentum = abs(momentum_5d)
            momentum_signal = min(0.8, abs_momentum / (threshold * 2))
            
            # Extreme RSI conditions
            rsi_signal = 0
            if rsi > 80 or rsi < 20:
                rsi_signal = 0.4
            elif rsi > 75 or rsi < 25:
                rsi_signal = 0.3
            elif rsi > 70 or rsi < 30:
                rsi_signal = 0.2
            
            # Volume spike
            volume_signal = min(0.3, max(0, (volume_ratio - 1.2) * 0.5))
            
            # Trend strength
            trend_signal = min(0.3, abs(trend_strength) / (threshold * 0.5))
            
            # Bollinger Bands extremes
            bb_signal = 0
            if bb_position > 95 or bb_position < 5:
                bb_signal = 0.3
            elif bb_position > 90 or bb_position < 10:
                bb_signal = 0.2
            
            # Combine signals
            total_signal = (volatility_signal * 0.35 + 
                          momentum_signal * 0.25 + 
                          rsi_signal * 0.15 + 
                          volume_signal * 0.1 + 
                          trend_signal * 0.1 + 
                          bb_signal * 0.05)
            
            # Threshold based on movement type
            sig_threshold = 0.25 + (threshold - 1) * 0.1
            
            prediction = 1 if total_signal > sig_threshold else 0
            confidence = min(0.95, max(0.55, 0.6 + abs(total_signal - sig_threshold) * 0.8))
        
        return prediction, confidence
    
    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    actual_returns: np.ndarray, predictions_prob: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Handle empty arrays
        if len(y_true) == 0 or len(y_pred) == 0:
            return {key: 0.0 for key in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score',
                                       'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown',
                                       'profit_factor', 'avg_win', 'avg_loss', 'volatility']}
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC score if probabilities provided
        if predictions_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc_score'] = roc_auc_score(y_true, predictions_prob)
            except:
                metrics['auc_score'] = 0.5
        else:
            metrics['auc_score'] = 0.5
        
        # Trading metrics
        strategy_returns = []
        for i, pred in enumerate(y_pred):
            if pred == 1:  # Predicted positive
                strategy_returns.append(actual_returns[i])
            else:  # Predicted negative - take opposite position
                strategy_returns.append(-actual_returns[i])
        
        strategy_returns = np.array(strategy_returns)
        
        # Remove any infinite or NaN values
        strategy_returns = strategy_returns[np.isfinite(strategy_returns)]
        
        if len(strategy_returns) == 0:
            metrics.update({
                'total_return': 0.0, 'sharpe_ratio': 0.0, 'win_rate': 0.0,
                'max_drawdown': 0.0, 'profit_factor': 0.0, 'avg_win': 0.0,
                'avg_loss': 0.0, 'volatility': 0.0
            })
            return metrics
        
        metrics['total_return'] = np.sum(strategy_returns)
        metrics['volatility'] = np.std(strategy_returns)
        
        # Sharpe ratio (annualized)
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = np.mean(strategy_returns) / metrics['volatility'] * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Win/Loss analysis
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        
        metrics['win_rate'] = len(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        metrics['avg_win'] = np.mean(wins) if len(wins) > 0 else 0
        metrics['avg_loss'] = np.mean(losses) if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Maximum drawdown
        try:
            cumulative = np.cumprod(1 + strategy_returns/100)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = np.min(drawdown)
        except:
            metrics['max_drawdown'] = 0
        
        return metrics
    
    def fixed_split_backtest(self, df: pd.DataFrame, movement_column: str, 
                           movement_type: str, train_ratio: float, 
                           run_id: int, stock_symbol: str = "") -> Optional[BacktestRun]:
        """Run backtest with fixed train-test split"""
        
        try:
            # Calculate split index
            split_idx = int(len(df) * train_ratio)
            
            # Ensure minimum test size
            min_test_size = 50
            if len(df) - split_idx < min_test_size:
                split_idx = len(df) - min_test_size
                if split_idx <= 0:
                    return None
            
            train_data = df.iloc[:split_idx].copy()
            test_data = df.iloc[split_idx:].copy()
            
            # Generate predictions for test period
            predictions = []
            confidences = []
            
            lookback_window = min(30, len(train_data) // 4)
            
            for i in range(lookback_window, len(test_data)):
                # Use combined train + test data up to current point (<= day t)
                # for technical analysis. This is causal: no row after t is used.
                historical_data = pd.concat([train_data, test_data.iloc[:i+1]])
                
                # Calculate technical indicators (trailing, causal)
                technical_data = self.technical_analyzer.calculate_indicators(historical_data)
                
                # Movement statistics: estimated from the causal history ending
                # at day t (training block plus already-seen test rows), never
                # from rows at or beyond the prediction target. This matches the
                # "Causality and Leakage Control" subsection of the paper.
                movement_stats = self.movement_analyzer.calculate_movement_stats(
                    historical_data, movement_column
                )
                
                # Generate prediction via the confidence-weighted LLM ensemble
                # (Eq. 5), or the rule baseline in ablation mode.
                pred, conf = self._predict(
                    stock_symbol, movement_type, technical_data, movement_stats
                )
                
                predictions.append(pred)
                confidences.append(conf)
            
            if len(predictions) == 0:
                return None
            
            # Get actual values for test period
            test_start_idx = lookback_window
            actual_movements = test_data.iloc[test_start_idx:test_start_idx + len(predictions)][movement_column].values
            actual_returns = test_data.iloc[test_start_idx:test_start_idx + len(predictions)]['daily_return'].values
            
            # Align arrays and remove NaN values
            min_len = min(len(predictions), len(actual_movements), len(actual_returns))
            predictions = np.array(predictions[:min_len])
            confidences = np.array(confidences[:min_len])
            actual_movements = actual_movements[:min_len]
            actual_returns = actual_returns[:min_len]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(actual_movements) | np.isnan(actual_returns))
            predictions = predictions[valid_mask]
            confidences = confidences[valid_mask]
            actual_movements = actual_movements[valid_mask]
            actual_returns = actual_returns[valid_mask]
            
            if len(predictions) == 0:
                return None
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics(
                actual_movements, predictions, actual_returns, confidences
            )
            
            # Create BacktestRun object
            return BacktestRun(
                run_id=run_id,
                stock_symbol="",  # Will be set by caller
                movement_type=movement_type,
                split_type="fixed",
                train_ratio=train_ratio,
                train_size=len(train_data),
                test_size=len(predictions),
                train_start_date=str(train_data['Date'].iloc[0].date()),
                train_end_date=str(train_data['Date'].iloc[-1].date()),
                test_start_date=str(test_data.iloc[lookback_window]['Date'].date()),
                test_end_date=str(test_data.iloc[lookback_window + len(predictions) - 1]['Date'].date()),
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                auc_score=metrics['auc_score'],
                total_return=metrics['total_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                win_rate=metrics['win_rate'],
                max_drawdown=metrics['max_drawdown'],
                profit_factor=metrics['profit_factor'],
                avg_win=metrics['avg_win'],
                avg_loss=metrics['avg_loss'],
                volatility=metrics['volatility'],
                total_predictions=len(predictions)
            )
            
        except Exception as e:
            logger.error(f"Error in fixed split backtest: {e}")
            return None
    
    def rolling_window_backtest(self, df: pd.DataFrame, movement_column: str,
                              movement_type: str, window_size_months: int,
                              step_size_months: int, run_id: int,
                              stock_symbol: str = "") -> List[BacktestRun]:
        """Run backtest with rolling window approach"""
        
        results = []
        
        try:
            # Convert months to approximate days
            window_size_days = window_size_months * 21  # ~21 trading days per month
            step_size_days = step_size_months * 21
            
            train_window_days = int(window_size_days * 0.8)  # 80% for training
            test_window_days = window_size_days - train_window_days
            
            # Ensure minimum window sizes
            train_window_days = max(train_window_days, 100)
            test_window_days = max(test_window_days, 25)
            
            start_idx = 0
            window_count = 0
            
            while start_idx + train_window_days + test_window_days < len(df):
                train_end_idx = start_idx + train_window_days
                test_end_idx = train_end_idx + test_window_days
                
                train_data = df.iloc[start_idx:train_end_idx].copy()
                test_data = df.iloc[train_end_idx:test_end_idx].copy()
                
                # Generate predictions
                predictions = []
                confidences = []
                
                lookback_window = min(20, len(train_data) // 4)
                
                for i in range(lookback_window, len(test_data)):
                    # Historical data up to day t (causal)
                    historical_data = pd.concat([train_data, test_data.iloc[:i+1]])
                    
                    # Calculate indicators (trailing, causal)
                    technical_data = self.technical_analyzer.calculate_indicators(historical_data)
                    # Movement stats from causal history ending at day t. Each
                    # rolling window estimates its statistics from a bounded
                    # block of recent history and forecasts strictly forward.
                    movement_stats = self.movement_analyzer.calculate_movement_stats(
                        historical_data, movement_column
                    )
                    
                    # Generate prediction via the LLM ensemble (Eq. 5)
                    pred, conf = self._predict(
                        stock_symbol, movement_type, technical_data, movement_stats
                    )
                    
                    predictions.append(pred)
                    confidences.append(conf)
                
                if len(predictions) > 0:
                    # Get actual values
                    test_start_idx = lookback_window
                    actual_movements = test_data.iloc[test_start_idx:test_start_idx + len(predictions)][movement_column].values
                    actual_returns = test_data.iloc[test_start_idx:test_start_idx + len(predictions)]['daily_return'].values
                    
                    # Align and clean data
                    min_len = min(len(predictions), len(actual_movements), len(actual_returns))
                    predictions = np.array(predictions[:min_len])
                    confidences = np.array(confidences[:min_len])
                    actual_movements = actual_movements[:min_len]
                    actual_returns = actual_returns[:min_len]
                    
                    valid_mask = ~(np.isnan(actual_movements) | np.isnan(actual_returns))
                    predictions = predictions[valid_mask]
                    confidences = confidences[valid_mask]
                    actual_movements = actual_movements[valid_mask]
                    actual_returns = actual_returns[valid_mask]
                    
                    if len(predictions) > 0:
                        # Calculate metrics
                        metrics = self.calculate_performance_metrics(
                            actual_movements, predictions, actual_returns, confidences
                        )
                        
                        # Create BacktestRun object
                        backtest_run = BacktestRun(
                            run_id=f"{run_id}_window_{window_count}",
                            stock_symbol="",  # Will be set by caller
                            movement_type=movement_type,
                            split_type=f"rolling_{window_size_months}m",
                            train_ratio=None,
                            train_size=len(train_data),
                            test_size=len(predictions),
                            train_start_date=str(train_data['Date'].iloc[0].date()),
                            train_end_date=str(train_data['Date'].iloc[-1].date()),
                            test_start_date=str(test_data.iloc[lookback_window]['Date'].date()),
                            test_end_date=str(test_data.iloc[lookback_window + len(predictions) - 1]['Date'].date()),
                            accuracy=metrics['accuracy'],
                            precision=metrics['precision'],
                            recall=metrics['recall'],
                            f1_score=metrics['f1_score'],
                            auc_score=metrics['auc_score'],
                            total_return=metrics['total_return'],
                            sharpe_ratio=metrics['sharpe_ratio'],
                            win_rate=metrics['win_rate'],
                            max_drawdown=metrics['max_drawdown'],
                            profit_factor=metrics['profit_factor'],
                            avg_win=metrics['avg_win'],
                            avg_loss=metrics['avg_loss'],
                            volatility=metrics['volatility'],
                            total_predictions=len(predictions)
                        )
                        
                        results.append(backtest_run)
                
                # Move window forward
                start_idx += step_size_days
                window_count += 1
                
                # Limit number of windows to prevent excessive computation
                if window_count >= 20:
                    break
            
        except Exception as e:
            logger.error(f"Error in rolling window backtest: {e}")
        
        return results
    
    def run_comprehensive_backtest(self, stock_symbols: List[str], 
                                 movement_types: List[str] = None,
                                 train_ratios: List[float] = None,
                                 num_runs_per_config: int = 20) -> List[BacktestRun]:
        """Run comprehensive backtest with multiple configurations"""
        
        if movement_types is None:
            movement_types = ['direction', 'sig_move_1pct', 'sig_move_2pct', 'sig_move_3pct']
        
        if train_ratios is None:
            train_ratios = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        
        all_results = []
        total_runs = len(stock_symbols) * len(movement_types) * (len(train_ratios) * num_runs_per_config + 20)  # +20 for rolling windows
        
        run_id = 0
        
        with tqdm(total=total_runs, desc="Running backtests") as pbar:
            for stock_symbol in stock_symbols:
                logger.info(f"Processing {stock_symbol}")
                
                # Get available datasets
                datasets = self.movement_analyzer.get_available_datasets(stock_symbol, self.data_directory)
                
                for movement_type in movement_types:
                    if movement_type not in datasets:
                        logger.warning(f"No data for {stock_symbol} - {movement_type}")
                        pbar.update(len(train_ratios) * num_runs_per_config + 20)
                        continue
                    
                    logger.info(f"  Testing {movement_type}")
                    
                    # Load and prepare data
                    file_path = datasets[movement_type]
                    df = pd.read_csv(file_path)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date').reset_index(drop=True)
                    
                    # Convert price columns
                    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
                    for col in price_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Calculate daily returns
                    df['daily_return'] = df['Close'].pct_change() * 100
                    
                    # Find movement column
                    movement_column = 'Direction' if 'Direction' in df.columns else 'Significant_Move'
                    
                    # Skip if insufficient data
                    if len(df) < 200:
                        logger.warning(f"Insufficient data for {stock_symbol} - {movement_type}")
                        pbar.update(len(train_ratios) * num_runs_per_config + 20)
                        continue
                    
                    # Fixed ratio backtests
                    # In ensemble mode the predictor is deterministic and cached,
                    # so re-running the same split ratio with jittered split points
                    # only re-issues identical (cached) queries. We therefore do a
                    # single run per ratio in ensemble mode; the Monte-Carlo
                    # variation is kept only for the stochastic rule baseline.
                    effective_runs = 1 if self.prediction_mode == "ensemble" else num_runs_per_config
                    for train_ratio in train_ratios:
                        for run_num in range(effective_runs):
                            # Add small random variation to split point for multiple runs
                            variation = np.random.uniform(-0.02, 0.02) if run_num > 0 else 0
                            adjusted_ratio = np.clip(train_ratio + variation, 0.6, 0.98)
                            
                            result = self.fixed_split_backtest(
                                df, movement_column, movement_type, adjusted_ratio,
                                run_id, stock_symbol=stock_symbol
                            )
                            
                            if result:
                                result.stock_symbol = stock_symbol
                                result.run_id = f"{run_id}_{stock_symbol}_{movement_type}_{train_ratio:.0%}_{run_num}"
                                all_results.append(result)
                            
                            run_id += 1
                            pbar.update(1)
                    # Keep the progress bar honest when we shortened the run grid.
                    if effective_runs < num_runs_per_config:
                        pbar.update(len(train_ratios) * (num_runs_per_config - effective_runs))
                    
                    # Rolling window backtests
                    rolling_configs = [
                        (12, 3),   # 12-month window, 3-month step
                        (18, 6),   # 18-month window, 6-month step
                        (24, 12),  # 24-month window, 12-month step
                    ]
                    
                    for window_months, step_months in rolling_configs:
                        rolling_results = self.rolling_window_backtest(
                            df, movement_column, movement_type, window_months,
                            step_months, run_id, stock_symbol=stock_symbol
                        )
                        
                        for result in rolling_results:
                            result.stock_symbol = stock_symbol
                            result.run_id = f"{run_id}_{stock_symbol}_{movement_type}_rolling_{window_months}m"
                            all_results.append(result)
                        
                        run_id += 1000  # Leave space for multiple rolling windows
                        pbar.update(7)  # Approximate update for rolling windows
                    
                    # Update remaining
                    remaining = 20 - len(rolling_configs) * 7
                    if remaining > 0:
                        pbar.update(remaining)
        
        logger.info(f"Completed {len(all_results)} backtest runs")
        return all_results

class AdvancedBacktestReporter:
    """Generate comprehensive reports with variable splits analysis"""
    
    def __init__(self, output_dir: str = "advanced_backtest_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_comprehensive_excel_report(self, results: List[BacktestRun]):
        """Create comprehensive Excel report with multiple sheets"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"comprehensive_backtest_results_{timestamp}.xlsx"
        
        # Convert results to DataFrame
        results_data = []
        for result in results:
            results_data.append({
                'Run_ID': result.run_id,
                'Stock_Symbol': result.stock_symbol,
                'Movement_Type': result.movement_type,
                'Split_Type': result.split_type,
                'Train_Ratio': result.train_ratio,
                'Train_Size': result.train_size,
                'Test_Size': result.test_size,
                'Train_Start_Date': result.train_start_date,
                'Train_End_Date': result.train_end_date,
                'Test_Start_Date': result.test_start_date,
                'Test_End_Date': result.test_end_date,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1_Score': result.f1_score,
                'AUC_Score': result.auc_score,
                'Total_Return': result.total_return,
                'Sharpe_Ratio': result.sharpe_ratio,
                'Win_Rate': result.win_rate,
                'Max_Drawdown': result.max_drawdown,
                'Profit_Factor': result.profit_factor,
                'Avg_Win': result.avg_win,
                'Avg_Loss': result.avg_loss,
                'Volatility': result.volatility,
                'Total_Predictions': result.total_predictions
            })
        
        df_all = pd.DataFrame(results_data)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # All results
            df_all.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Summary by train ratio
            fixed_results = df_all[df_all['Split_Type'] == 'fixed']
            if len(fixed_results) > 0:
                ratio_summary = fixed_results.groupby(['Train_Ratio', 'Movement_Type']).agg({
                    'Accuracy': ['mean', 'std', 'min', 'max'],
                    'F1_Score': ['mean', 'std', 'min', 'max'],
                    'Sharpe_Ratio': ['mean', 'std', 'min', 'max'],
                    'Total_Return': ['mean', 'std', 'min', 'max'],
                    'Win_Rate': ['mean', 'std', 'min', 'max']
                }).round(4)
                
                # Flatten column names
                ratio_summary.columns = ['_'.join(col).strip() for col in ratio_summary.columns]
                ratio_summary.reset_index().to_excel(writer, sheet_name='Train_Ratio_Summary', index=False)
            
            # Summary by stock and movement type
            stock_summary = df_all.groupby(['Stock_Symbol', 'Movement_Type']).agg({
                'Accuracy': ['mean', 'std', 'count'],
                'F1_Score': ['mean', 'std'],
                'Sharpe_Ratio': ['mean', 'std'],
                'Total_Return': ['mean', 'std'],
                'Win_Rate': ['mean', 'std']
            }).round(4)
            
            stock_summary.columns = ['_'.join(col).strip() for col in stock_summary.columns]
            stock_summary.reset_index().to_excel(writer, sheet_name='Stock_Movement_Summary', index=False)
            
            # Split type comparison
            split_summary = df_all.groupby(['Split_Type', 'Movement_Type']).agg({
                'Accuracy': ['mean', 'std', 'count'],
                'F1_Score': ['mean', 'std'],
                'Sharpe_Ratio': ['mean', 'std'],
                'Total_Return': ['mean', 'std']
            }).round(4)
            
            split_summary.columns = ['_'.join(col).strip() for col in split_summary.columns]
            split_summary.reset_index().to_excel(writer, sheet_name='Split_Type_Comparison', index=False)
            
            # Best performers
            best_performers = []
            
            for metric in ['Accuracy', 'F1_Score', 'Sharpe_Ratio', 'Total_Return']:
                best_idx = df_all[metric].idxmax()
                if pd.notna(best_idx):
                    best_run = df_all.loc[best_idx]
                    best_performers.append({
                        'Metric': metric,
                        'Value': best_run[metric],
                        'Stock': best_run['Stock_Symbol'],
                        'Movement_Type': best_run['Movement_Type'],
                        'Split_Type': best_run['Split_Type'],
                        'Train_Ratio': best_run['Train_Ratio']
                    })
            
            pd.DataFrame(best_performers).to_excel(writer, sheet_name='Best_Performers', index=False)
            
            # Detailed statistics by train ratio
            if len(fixed_results) > 0:
                for ratio in sorted(fixed_results['Train_Ratio'].unique()):
                    if pd.notna(ratio):
                        ratio_data = fixed_results[fixed_results['Train_Ratio'] == ratio]
                        sheet_name = f"Ratio_{int(ratio*100)}_Detail"
                        ratio_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Comprehensive Excel report saved: {filename}")
        return filename
    
    def generate_summary_report(self, results: List[BacktestRun]) -> str:
        """Generate summary text report"""
        
        lines = []
        lines.append("📊 COMPREHENSIVE BACKTEST ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Analysis Date: {datetime.now().isoformat()}")
        lines.append(f"Total Runs: {len(results)}")
        lines.append("")
        
        if len(results) == 0:
            lines.append("No results to analyze.")
            return "\n".join(lines)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'Stock': r.stock_symbol,
            'Movement_Type': r.movement_type,
            'Split_Type': r.split_type,
            'Train_Ratio': r.train_ratio,
            'Accuracy': r.accuracy,
            'F1_Score': r.f1_score,
            'Sharpe_Ratio': r.sharpe_ratio,
            'Total_Return': r.total_return,
            'Win_Rate': r.win_rate
        } for r in results])
        
        # Overall statistics
        lines.append("📈 OVERALL PERFORMANCE STATISTICS:")
        lines.append("-" * 50)
        for metric in ['Accuracy', 'F1_Score', 'Sharpe_Ratio', 'Win_Rate']:
            lines.append(f"{metric}:")
            lines.append(f"  Mean: {df[metric].mean():.3f}")
            lines.append(f"  Std:  {df[metric].std():.3f}")
            lines.append(f"  Min:  {df[metric].min():.3f}")
            lines.append(f"  Max:  {df[metric].max():.3f}")
            lines.append("")
        
        # Train ratio analysis
        fixed_results = df[df['Split_Type'] == 'fixed']
        if len(fixed_results) > 0:
            lines.append("🎯 TRAIN-TEST RATIO ANALYSIS:")
            lines.append("-" * 50)
            
            ratio_stats = fixed_results.groupby('Train_Ratio').agg({
                'Accuracy': ['mean', 'std'],
                'F1_Score': ['mean', 'std'],
                'Sharpe_Ratio': ['mean', 'std']
            }).round(3)
            
            for ratio in sorted(fixed_results['Train_Ratio'].unique()):
                if pd.notna(ratio):
                    lines.append(f"\nTrain Ratio {ratio:.0%}:")
                    ratio_data = fixed_results[fixed_results['Train_Ratio'] == ratio]
                    lines.append(f"  Runs: {len(ratio_data)}")
                    lines.append(f"  Accuracy:  {ratio_data['Accuracy'].mean():.3f} ± {ratio_data['Accuracy'].std():.3f}")
                    lines.append(f"  F1-Score:  {ratio_data['F1_Score'].mean():.3f} ± {ratio_data['F1_Score'].std():.3f}")
                    lines.append(f"  Sharpe:    {ratio_data['Sharpe_Ratio'].mean():.3f} ± {ratio_data['Sharpe_Ratio'].std():.3f}")
        
        # Movement type analysis
        lines.append("\n📊 MOVEMENT TYPE ANALYSIS:")
        lines.append("-" * 50)
        
        movement_stats = df.groupby('Movement_Type').agg({
            'Accuracy': ['mean', 'std', 'count'],
            'F1_Score': ['mean', 'std']
        }).round(3)
        
        for movement_type in df['Movement_Type'].unique():
            movement_data = df[df['Movement_Type'] == movement_type]
            lines.append(f"\n{movement_type.upper()}:")
            lines.append(f"  Runs: {len(movement_data)}")
            lines.append(f"  Accuracy:  {movement_data['Accuracy'].mean():.3f} ± {movement_data['Accuracy'].std():.3f}")
            lines.append(f"  F1-Score:  {movement_data['F1_Score'].mean():.3f} ± {movement_data['F1_Score'].std():.3f}")
        
        # Best configurations
        lines.append("\n🏆 BEST CONFIGURATIONS:")
        lines.append("-" * 50)
        
        for metric in ['Accuracy', 'F1_Score', 'Sharpe_Ratio']:
            best_idx = df[metric].idxmax()
            if pd.notna(best_idx):
                best_run = df.loc[best_idx]
                lines.append(f"\nBest {metric}: {best_run[metric]:.3f}")
                lines.append(f"  Stock: {best_run['Stock']}")
                lines.append(f"  Movement: {best_run['Movement_Type']}")
                lines.append(f"  Split: {best_run['Split_Type']}")
                if pd.notna(best_run['Train_Ratio']):
                    lines.append(f"  Ratio: {best_run['Train_Ratio']:.0%}")
        
        # Save report
        report_text = "\n".join(lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"comprehensive_backtest_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Summary report saved: {report_file}")
        return report_text

# Main execution functions
def run_comprehensive_analysis(prediction_mode: str = "ensemble", tau: float = 0.5,
                               use_tiered_weights: bool = False):
    """Run comprehensive backtest analysis"""
    
    engine = AdvancedBacktestEngine("data", prediction_mode=prediction_mode,
                                    tau=tau, use_tiered_weights=use_tiered_weights)
    reporter = AdvancedBacktestReporter()
    
    print("🚀 Advanced Comprehensive Backtest Analysis")
    print("=" * 60)
    print("Configuration:")
    print(f"- Prediction mode: {prediction_mode.upper()}")
    print("- Train-Test Ratios: 70-30, 75-25, 80-20, 85-15, 90-10, 95-5")
    print("- Rolling windows: 12m, 18m, 24m")
    print("- Stocks: NIFTY50, NIFTYBANK, NIFTYIT")
    print("- Movement types: Direction, 1%, 2%, 3%")
    print("")
    
    try:
        # Run comprehensive backtest
        stocks = ['NIFTY50', 'NIFTYBANK', 'NIFTYIT']
        train_ratios = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        
        results = engine.run_comprehensive_backtest(
            stock_symbols=stocks,
            train_ratios=train_ratios,
            num_runs_per_config=20
        )
        
        if len(results) == 0:
            print("❌ No results generated")
            return
        
        print(f"\n✅ Generated {len(results)} backtest runs")
        
        # Generate reports
        print("📊 Generating comprehensive Excel report...")
        excel_file = reporter.create_comprehensive_excel_report(results)
        
        print("📝 Generating summary report...")
        summary_report = reporter.generate_summary_report(results)
        
        print("\n" + summary_report)
        print(f"\n📁 Detailed results saved to: {excel_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Comprehensive analysis failed: {e}", exc_info=True)

def run_quick_test(prediction_mode: str = "ensemble", tau: float = 0.5,
                   use_tiered_weights: bool = False):
    """Run quick test with fewer configurations"""
    
    engine = AdvancedBacktestEngine("data", prediction_mode=prediction_mode,
                                    tau=tau, use_tiered_weights=use_tiered_weights)
    reporter = AdvancedBacktestReporter()
    
    print("🚀 Quick Advanced Backtest Test")
    print(f"   (prediction mode: {prediction_mode.upper()})")
    print("=" * 40)
    
    try:
        # Quick test with fewer runs
        results = engine.run_comprehensive_backtest(
            stock_symbols=['NIFTY50'],
            train_ratios=[0.80, 0.90],
            num_runs_per_config=5
        )
        
        if len(results) > 0:
            print(f"✅ Generated {len(results)} test runs")
            
            # Show sample results
            for i, result in enumerate(results[:3]):
                print(f"\nRun {i+1}: {result.stock_symbol} - {result.movement_type}")
                print(f"  Split: {result.split_type} (ratio: {result.train_ratio})")
                print(f"  Accuracy: {result.accuracy:.3f}")
                print(f"  F1-Score: {result.f1_score:.3f}")
                print(f"  Sharpe: {result.sharpe_ratio:.3f}")
            
            # Generate Excel report
            excel_file = reporter.create_comprehensive_excel_report(results)
            print(f"\n📁 Test results saved to: {excel_file}")
        else:
            print("❌ No results generated")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Advanced LLM-ensemble backtesting system")
    parser.add_argument("--scope", choices=["quick", "full"], default="quick",
                        help="quick: NIFTY50 only, 2 ratios; full: all indices/ratios")
    parser.add_argument("--mode", choices=["ensemble", "rule"], default="ensemble",
                        help="ensemble: confidence-weighted LLM ensemble (Eq. 5, "
                             "reproduces the paper); rule: legacy technical-rule "
                             "ablation baseline")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="ensemble decision threshold tau in Eq. (5)")
    parser.add_argument("--tiered-weights", action="store_true",
                        help="use heterogeneous reliability weights w_i "
                             "(default: equal weights => confidence-weighted voting)")
    args = parser.parse_args()

    print("🧪 Advanced LLM Stock Prediction Backtesting System")
    print(f"   scope={args.scope}  mode={args.mode}  tau={args.tau}  "
          f"tiered_weights={args.tiered_weights}")

    if args.scope == "full":
        run_comprehensive_analysis(prediction_mode=args.mode, tau=args.tau,
                                   use_tiered_weights=args.tiered_weights)
    else:
        run_quick_test(prediction_mode=args.mode, tau=args.tau,
                       use_tiered_weights=args.tiered_weights)
