# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from datetime import datetime

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Load the data
print("Loading data...")
data = pd.read_csv('data\processed\merged_data.csv', parse_dates=['Date'], index_col='Date')
print(f"Data loaded with shape: {data.shape}")

# Define asset classes
asset_classes = {
    'us_large_cap': ['SPY_Close', 'QQQ_Close'],
    'us_tech': ['MSFT_Close', 'AMZN_Close', 'GOOGL_Close', 'TSLA_Close'],
    'commodities': ['GC=F_Close', 'CL=F_Close', 'SI=F_Close'],
    'real_estate': ['VNQ_Close', 'SCHH_Close', 'IYR_Close'],
    'crypto_major': ['BTC_Close', 'ETH_Close'],
    'crypto_alt': ['LTC_Close']
}

class PortfolioAllocator:
    def __init__(self, data, target_col='SPY_Close'):
        self.data = data
        self.target_col = target_col
        self.scaler = StandardScaler()
        
        # Enhanced RandomForest configuration
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            bootstrap=True,
            max_samples=0.8,
            verbose=1
        )
        print("Portfolio Allocator initialized")

    def create_features(self, data):
        """Create sophisticated features for the model."""
        features = pd.DataFrame(index=data.index)
        
        for column in data.columns:
            prices = data[column]
            returns = prices.pct_change()
            
            # Price momentum features
            features[f'{column}_ret_1d'] = returns
            features[f'{column}_ret_5d'] = returns.rolling(5).mean()
            features[f'{column}_ret_20d'] = returns.rolling(20).mean()
            
            # Volatility features
            features[f'{column}_vol_20d'] = returns.rolling(20).std()
            features[f'{column}_vol_60d'] = returns.rolling(60).std()
            
            # Technical indicators
            features[f'{column}_rsi'] = self._calculate_rsi(prices)
            features[f'{column}_ma_cross'] = (prices.rolling(10).mean() / 
                                            prices.rolling(30).mean() - 1)
            
            # Momentum and mean reversion
            features[f'{column}_mom_20d'] = prices.pct_change(20)
            features[f'{column}_mom_60d'] = prices.pct_change(60)
            
            # Volatility regime
            features[f'{column}_vol_regime'] = (returns.rolling(20).std() / 
                                              returns.rolling(60).std())
        
        return features.dropna()

    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI technical indicator."""
        returns = prices.diff()
        pos = returns.clip(lower=0).ewm(span=periods).mean()
        neg = (-returns.clip(upper=0)).ewm(span=periods).mean()
        rs = pos/neg
        return 100 - (100/(1 + rs))

    def prepare_ml_data(self):
        """Prepare data for ML models."""
        target = self.data[self.target_col].pct_change().shift(-1).dropna()
        features = self.create_features(self.data.drop(columns=[self.target_col]))
        
        valid_indices = features.index.intersection(target.index)
        features = features.loc[valid_indices]
        target = target.loc[valid_indices]
        
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features, target.values, features

    def train_model(self):
        """Train the Random Forest model."""
        print("Training Random Forest model...")
        start_time = time.time()
        
        scaled_features, target, features = self.prepare_ml_data()
        self.model.fit(scaled_features, target)
        
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        return features

    def plot_feature_importances(self, features, top_n=20):
        """Plot feature importances."""
        importances = pd.Series(
            self.model.feature_importances_,
            index=features.columns
        ).sort_values(ascending=True)[-top_n:]
        
        plt.figure(figsize=(12, 8))
        importances.plot(kind='barh')
        plt.title('Top Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()


    def calculate_yearly_weights(self):
        """Calculate dynamic weights for each asset class by year."""
        print("\nCalculating yearly weights...")
        
        yearly_weights = {}
        yearly_metrics = {}  # Store metrics for visualization
        
        for year in range(2013, 2024):
            year_data = self.data[self.data.index.year == year]
            
            if not year_data.empty:
                weights = {}
                metrics = {}
                
                for asset_class, assets in asset_classes.items():
                    # Get asset data
                    asset_data = year_data[assets]
                    returns = asset_data.pct_change()
                    
                    # Calculate comprehensive metrics
                    avg_returns = returns.mean()
                    risk = returns.std()
                    
                    # 1. Sharpe ratio
                    sharpe = (avg_returns / (risk + 1e-6)).mean()
                    
                    # 2. Momentum (exponentially weighted)
                    momentum = returns.ewm(span=60).mean().iloc[-1] if len(returns) > 60 else returns.mean()
                    momentum = momentum.mean()
                    
                    # 3. Volatility score
                    vol_score = 1 / (risk.mean() + 1e-6)
                    
                    # 4. Trend strength
                    prices = asset_data.mean(axis=1)
                    trend = (prices.iloc[-1] / prices.iloc[0] - 1) if len(prices) > 1 else 0
                    
                    # 5. Drawdown protection
                    rolling_max = prices.rolling(window=252, min_periods=1).max()
                    drawdown = (prices - rolling_max) / rolling_max
                    max_drawdown = drawdown.min()
                    
                    # Combine scores with weights
                    combined_score = (
                        0.35 * sharpe +
                        0.25 * momentum +
                        0.15 * vol_score +
                        0.15 * trend +
                        0.10 * (1 + max_drawdown)  # Convert drawdown to positive factor
                    )
                    
                    weights[asset_class] = max(0, combined_score)
                    metrics[asset_class] = {
                        'sharpe': sharpe,
                        'momentum': momentum,
                        'volatility': risk.mean(),
                        'trend': trend,
                        'max_drawdown': max_drawdown
                    }
                
                # Handle zero weights
                if sum(weights.values()) == 0:
                    weights = {k: 1/len(weights) for k in weights}
                
                # Apply dynamic constraints based on market conditions
                bull_markets = [2017, 2020, 2021]
                bear_markets = [2018, 2022]
                
                for asset_class in weights:
                    base_weight = weights[asset_class]
                    
                    if year in bull_markets:
                        if asset_class in ['us_tech', 'crypto_major']:
                            min_alloc, max_alloc = 0.10, 0.40
                        else:
                            min_alloc, max_alloc = 0.05, 0.30
                    elif year in bear_markets:
                        if asset_class in ['commodities', 'us_large_cap']:
                            min_alloc, max_alloc = 0.15, 0.40
                        else:
                            min_alloc, max_alloc = 0.05, 0.25
                    else:
                        min_alloc, max_alloc = 0.10, 0.35
                    
                    weights[asset_class] = max(min_alloc, min(base_weight, max_alloc))
                
                # Normalize weights
                total_weight = sum(weights.values())
                weights = {k: v / total_weight for k, v in weights.items()}
                
                yearly_weights[year] = weights
                yearly_metrics[year] = metrics
                
                print(f"\nYear {year} allocations:")
                for asset_class, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                    print(f"{asset_class:<15}: {weight:.2%}")
        
        return yearly_weights, yearly_metrics

    def create_visualizations(self, yearly_weights, yearly_metrics):
        """Create comprehensive visualizations of the portfolio allocation and performance."""
        
        # 1. Portfolio Performance vs S&P 500
        portfolio_returns = self.calculate_portfolio_performance(yearly_weights)
        benchmark_returns = (1 + self.data['^GSPC_Close'].pct_change()).cumprod()
        
        plt.figure(figsize=(15, 8))
        plt.plot(portfolio_returns.index, portfolio_returns['Cumulative'], 
                 label='Portfolio', linewidth=2)
        plt.plot(self.data.index, benchmark_returns, 
                 label='S&P 500', linewidth=2, linestyle='--')
        plt.title('Portfolio Performance vs S&P 500')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. Stacked area chart of allocations over time
        plt.figure(figsize=(15, 8))
        df_weights = pd.DataFrame(yearly_weights).T
        df_weights.plot(kind='area', stacked=True, colormap='viridis')
        plt.title('Portfolio Allocation Over Time')
        plt.xlabel('Year')
        plt.ylabel('Allocation Percentage')
        plt.legend(title='Asset Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 3. Heatmap of allocations
        plt.figure(figsize=(15, 8))
        sns.heatmap(df_weights.T, annot=True, fmt='.2%', cmap='YlOrRd', center=0.2)
        plt.title('Asset Allocation Heatmap')
        plt.tight_layout()
        plt.show()

        # 4. Performance metrics visualization
        metrics_data = {}
        for year in yearly_metrics:
            for asset_class in yearly_metrics[year]:
                if asset_class not in metrics_data:
                    metrics_data[asset_class] = {}
                metrics_data[asset_class][year] = yearly_metrics[year][asset_class]

        for metric in ['sharpe', 'momentum', 'volatility', 'trend', 'max_drawdown']:
            plt.figure(figsize=(15, 6))
            for asset_class in metrics_data:
                years = list(metrics_data[asset_class].keys())
                values = [metrics_data[asset_class][year][metric] for year in years]
                plt.plot(years, values, marker='o', label=asset_class)
            
            plt.title(f'{metric.capitalize()} Over Time')
            plt.xlabel('Year')
            plt.ylabel(metric.capitalize())
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 5. Portfolio Performance
        portfolio_returns = self.calculate_portfolio_performance(yearly_weights)
        
        # 6. Risk-Return Scatter Plot
        plt.figure(figsize=(12, 8))
        for asset_class in asset_classes:
            returns = []
            risks = []
            for year in yearly_metrics:
                metrics = yearly_metrics[year][asset_class]
                returns.append(metrics['sharpe'])
                risks.append(metrics['volatility'])
            plt.scatter(risks, returns, label=asset_class, alpha=0.6, s=100)
        
        plt.xlabel('Risk (Volatility)')
        plt.ylabel('Risk-Adjusted Return (Sharpe)')
        plt.title('Risk-Return Profile by Asset Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def calculate_portfolio_performance(self, yearly_weights):
        """Calculate portfolio performance."""
        portfolio_returns = pd.DataFrame(index=self.data.index)
        
        # Calculate portfolio returns
        for year in yearly_weights:
            year_data = self.data[self.data.index.year == year]
            weights = yearly_weights[year]
            
            year_returns = pd.DataFrame(index=year_data.index)
            for asset_class, assets in asset_classes.items():
                asset_returns = year_data[assets].pct_change().mean(axis=1) * weights[asset_class]
                year_returns[asset_class] = asset_returns
            
            portfolio_returns.loc[year_data.index, 'Portfolio'] = year_returns.sum(axis=1)
        
        # Calculate cumulative returns
        portfolio_returns['Cumulative'] = (1 + portfolio_returns['Portfolio']).cumprod()
        
        return portfolio_returns

    def calculate_performance_metrics(self, portfolio_returns):
        """Calculate comprehensive performance metrics."""
        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol
        
        # Calculate drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate monthly returns
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Calculate win rate
        winning_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        win_rate = winning_months / total_months
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_std if downside_std != 0 else np.nan
        
        # Calculate rolling Sharpe ratio (12-month window)
        rolling_sharpe = portfolio_returns.rolling(window=252).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if len(x) > 1 else np.nan
        )
        
        metrics = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Best Month': monthly_returns.max(),
            'Worst Month': monthly_returns.min(),
            'Rolling Sharpe': rolling_sharpe
        }
        
        return metrics

    def display_performance_metrics(metrics):
        """Display performance metrics with enhanced visualization."""
        # Print metrics
        print("\nPortfolio Performance Metrics:")
        print("=" * 40)
        for metric, value in metrics.items():
            if metric != 'Rolling Sharpe':
                if isinstance(value, float):
                    print(f"{metric:<20}: {value:>10.2%}")
                else:
                    print(f"{metric:<20}: {value:>10.2f}")
        
        # Plot rolling metrics
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Rolling Sharpe ratio
        metrics['Rolling Sharpe'].plot(ax=axes[0])
        axes[0].set_title('Rolling Sharpe Ratio (12-month window)')
        axes[0].grid(True)
        
        # Monthly returns distribution
        monthly_returns = metrics.get('Monthly Returns', pd.Series())
        sns.histplot(monthly_returns, kde=True, ax=axes[1])
        axes[1].set_title('Distribution of Monthly Returns')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    try:
        # Load data
        print("Loading data...")
        data = pd.read_csv('data\processed\merged_data.csv', parse_dates=['Date'], index_col='Date')
        print(f"Data loaded with shape: {data.shape}")
        
        # Initialize allocator
        allocator = PortfolioAllocator(data)
        
        # Train model and get features
        print("\nTraining model...")
        features = allocator.train_model()
        
        # Plot feature importances
        print("\nPlotting feature importances...")
        allocator.plot_feature_importances(features)
        
        # Calculate weights and metrics using the class method
        print("\nCalculating portfolio weights...")
        yearly_weights, yearly_metrics = allocator.calculate_yearly_weights()  # No data parameter needed
        
        # Create visualizations using the class method
        print("\nCreating visualizations...")
        allocator.create_visualizations(yearly_weights, yearly_metrics)  # No data parameter needed
        
        # Calculate performance using the class method
        print("\nCalculating performance...")
        portfolio_returns = allocator.calculate_portfolio_performance(yearly_weights) 
        
        # Save results
        results_df = pd.DataFrame(yearly_weights).T
        results_df.to_csv('portfolio_allocations.csv')
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to 'portfolio_allocations.csv'")
        
        # Additional analysis outputs
        print("\nGenerating performance report...")
        
        # Calculate total return using portfolio returns
        total_return = portfolio_returns['Cumulative'].iloc[-1] - 1  # Changed to use cumulative returns
        print(f"Total Portfolio Return: {total_return:.2%}")
        
        # Calculate average turnover
        turnover = np.abs(results_df.diff()).mean().mean()
        print(f"Average Annual Turnover: {turnover:.2%}")
        
        # Display asset class correlations
        print("\nAsset Class Correlations:")
        returns_data = pd.DataFrame()
        for asset_class, assets in asset_classes.items():
            returns_data[asset_class] = data[assets].pct_change().mean(axis=1)
        
        correlation_matrix = returns_data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Asset Class Correlations')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise