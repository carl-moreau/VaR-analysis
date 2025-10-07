# import
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm, skew, kurtosis
from math import floor, ceil
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from datetime import datetime
from curl_cffi import requests
session = requests.Session(impersonate="chrome")


class Asset:
    """
    Represents a financial asset with historical price data and performance metrics.
    
    This class encapsulates all functionality related to individual financial assets,
    including data retrieval, return calculations, and statistical analysis. It serves
    as the building block for portfolio construction and analysis.
    
    Attributes:
        name (str): Name of the asset
        ticker (str): Financial ticker symbol for data retrieval
        start (datetime): Start date for historical data
        end (datetime): End date for historical data
        prices (pd.Series): Historical closing prices
        simple_returns (pd.Series): Daily simple returns
        mu (float): Mean daily return
        sigma (float): Daily return volatility (standard deviation)
    """

    def __init__(self, name: str, ticker: str, start: datetime = datetime(2000,1,1), end: datetime = datetime.today(),
                 trading_days: int = 252):
        """
        Initialize an Asset instance with specified parameters.
        
        Downloads historical price data and calculates basic return statistics
        for the specified time period.
        
        Args:
            name (str): Asset name (e.g., "Apple")
            ticker (str): Yahoo Finance ticker symbol (e.g., "AAPL")
            start (datetime, optional): Start date for data retrieval. 
                Defaults to datetime(2000,1,1).
            end (datetime, optional): End date for data retrieval.
                Defaults to current date.
        
        Raises:
            ConnectionError: If data download fails due to network issues
                or invalid ticker symbol.
        
        Example:
            asset = Asset("Apple", "AAPL", datetime(2020,1,1))
        """
        self.name = name
        self.ticker = ticker
        self.start = start
        self.end = end
        self.prices = self._download_data()
        self.simple_returns = self.prices.pct_change().dropna().squeeze()
        self.mu = self.simple_returns.mean()   # daily
        self.sigma = self.simple_returns.std()
    
    def __str__(self):
        """
        Return a formatted string representation of the asset's performance metrics.
        
        Provides a comprehensive summary including total return, annualized metrics,
        Sharpe ratio, and maximum drawdown for the asset's historical period.
        
        Returns:
            str: Multi-line formatted string containing:
                - Asset name and ticker
                - Analysis period
                - Total and annualized returns
                - Volatility metrics
                - Risk-adjusted performance measures
        """
        txt = f'Asset name : {self.name} ({self.ticker})\n'
        txt += f"Asset Historical Performance from {self.simple_returns.index[0].strftime('%Y-%m-%d')} to {self.simple_returns.index[-1].strftime('%Y-%m-%d')}\n"
        stats = self.compute_statistics()  
        txt += (
            f"Total Return         : {stats['total_return']:.2%}\n"
            f"Annualized Return    : {stats['annual_return']:.2%}\n"
            f"Annualized Volatility: {stats['annual_volatility']:.2%}\n"
            f"Sharpe Ratio (r_f={stats['risk_free_rate']:.2%}): {stats['sharpe_ratio']:.2f}\n"
            f"Max Drawdown         : {stats['max_drawdown']:.2%}\n"
        )
        return txt

    def _download_data(self):
        """
        Download historical price data from Yahoo Finance.
        
        Private method that retrieves closing prices for the specified
        ticker and date range. Uses curl_cffi session for improved reliability.
        
        Returns:
            pd.Series: Time series of daily closing prices with datetime index.
        
        Raises:
            ConnectionError: When data retrieval fails due to network issues,
                invalid ticker, or API limitations.
        
        Note:
            This method is called automatically during initialization and should
            not be called directly by users.
        """
        try:
            df = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=False, progress=False, session=session)
            return df['Close'].dropna().squeeze()
        except:
            raise ConnectionError("Failed to download data")
    
    def compute_statistics(self, risk_free_rate=0.02):
        """
        Calculate comprehensive performance statistics for the asset.
        
        Computes key financial metrics including returns, volatility, risk-adjusted
        performance, and drawdown analysis.
        
        Args:
            risk_free_rate (float, optional): Annual risk-free rate for Sharpe ratio
                calculation. Defaults to 0.02 (2%).
        
        Returns:
            dict: Dictionary containing the following keys:
                - 'total_return' (float): Cumulative return over the period
                - 'annual_return' (float): Annualized return (geometric mean)
                - 'annual_volatility' (float): Annualized volatility
                - 'sharpe_ratio' (float): Risk-adjusted return metric
                - 'risk_free_rate' (float): Input risk-free rate
                - 'max_drawdown' (float): Maximum peak-to-trough decline
        """
        total_return = (self.prices.iloc[-1] / self.prices.iloc[0]) - 1
        annual_return = (1 + self.simple_returns.mean())**self.trading_days - 1
        annual_vol = self.simple_returns.std() * np.sqrt(self.trading_days)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else np.nan        
        
        cum_returns = (1 + self.simple_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'risk_free_rate': risk_free_rate,
            'max_drawdown': max_drawdown
        }
    
    def plot_prices(self):
        """
        Display a line plot of the asset's historical price evolution.
        
        Creates a matplotlib time series plot showing the asset's price movement
        over the specified period. Useful for visual trend analysis.
        
        Returns:
            None: Displays plot directly using plt.show()
        """
        self.prices.plot()
        plt.title(f'{self.name} ({self.ticker}) prices')
        plt.show()

    def plot_returns(self):
        """
        Display a histogram of the asset's daily returns distribution.
        
        Creates a histogram with 100 bins showing the frequency distribution
        of daily returns. Useful for assessing return normality and tail behavior.
        
        Returns:
            None: Displays plot directly using plt.show()
        """
        plt.hist(self.simple_returns, bins=100, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        plt.title("Returns histogram")
        plt.show()


class Portfolio:
    """
    Represents a portfolio of financial assets.
    
    This class provides multiple Value at Risk (VaR) calculation methods for risk management. 
    Supports both equal-weighted and custom-weighted portfolio construction.
    
    Attributes:
        name (str): Portfolio identifier name
        assets (list[Asset]): List of Asset objects in the portfolio
        weights (np.array): Asset allocation weights (must sum to 1.0)
        log_port_returns (pd.Series): Portfolio log returns time series
        initial_value (float): Initial portfolio value for analysis
        conf_level (float): Confidence level for VaR calculations (e.g., 95.0)
        time_horizon (int): Time horizon in days for risk measures
        mu (float): Mean daily portfolio log return
        sigma (float): Daily portfolio log return volatility
    """

    def __init__(self, name, assets : list[Asset], weights: list[float] = None, initial_value: float = 1000000,
                 conf_level: float = 95.00, time_horizon: int = 1, trading_days: int = 252):
        """
        Initialize a Portfolio instance with specified assets and parameters.
        
        Creates a portfolio with given assets, validates weights, and computes
        portfolio-level returns.
        
        Args:
            name (str): Portfolio identifier name
            assets (list[Asset]): List of Asset objects to include
            weights (list or np.array): Asset allocation weights (must sum to 1.0)
            initial_value (float): Initial portfolio value in currency units
            conf_level (float): Confidence level for VaR (e.g., 95.0 for 95%)
            time_horizon (int): Risk measurement horizon in days
        
        Raises:
            ValueError: If weights don't sum to 1.0 or length doesn't match assets
        """
        self.name = name
        self.assets = assets
        self.weights = self._validate_weights(weights)
        self.initial_value = initial_value
        self.conf_level = conf_level
        self.time_horizon = time_horizon
        self.trading_days = trading_days
        self.log_port_returns = self._compute_portfolio_returns()
        self.mu = self.log_port_returns.mean()   # daily
        self.sigma = self.log_port_returns.std()

    def __str__(self):
        """
        Return a comprehensive formatted summary of the portfolio.
        
        Provides detailed information about portfolio composition, individual
        asset statistics, and overall portfolio performance metrics.
        
        Returns:
            str: Multi-line formatted string containing:
                - Portfolio name and parameters
                - Individual asset details and weights
                - Portfolio performance statistics
                - Risk metrics and drawdown analysis
        """
        txt = f"\nPortfolio Summary: {self.name}\n"
        txt += f"Initial Value: ${self.initial_value:,.0f}\n"
        txt += f"Confidence Level: {self.conf_level:.2f}% | Time Horizon: {self.time_horizon} days\n"
        txt += "-" * 100 + "\n"
        txt += "Assets:\n"
        for i, asset in enumerate(self.assets):
            txt += (
                f"  + {asset.name:<12} ({asset.ticker}) | "
                f"Weight: {self.weights[i]:>6.2%} | "
                f"Daily μ: {asset.mu:>6.3%} | "
                f"Daily σ: {asset.sigma:>6.3%} | "
                f"Ann. μ: {asset.mu * self.trading_days:>6.2%} | "
                f"Ann. σ: {asset.sigma * np.sqrt(self.trading_days):>6.2%}\n"
            )

        txt += f"\nPortfolio Historical Performance from {self.log_port_returns.index[0].strftime('%Y-%m-%d')} to {self.log_port_returns.index[-1].strftime('%Y-%m-%d')}\n"
        txt += "-" * 100 + "\n"

        stats = self.compute_statistics()
        txt += (
            f"Final Value         : ${stats['final_value']:,.2f}\n"
            f"Total Return         : {stats['total_return']:.2%}\n"
            f"Annualized Return    : {stats['annual_return']:.2%}\n"
            f"Annualized Volatility: {stats['annual_volatility']:.2%}\n"
            f"Sharpe Ratio (r_f={stats['risk_free_rate']:.2%}): {stats['sharpe_ratio']:.2f}\n"
            f"Max Drawdown         : {stats['max_drawdown']:.2%}\n"
        )
        return txt

    def _validate_weights(self, weights):
        """
        Validate and normalize portfolio weights.
        
        Private method that ensures weights are properly formatted and sum to 1.0.
        If no weights provided, creates equal-weight allocation.
        
        Args:
            weights (list, np.array, or None): Asset allocation weights
        
        Returns:
            np.array: Validated weights array
        
        Raises:
            ValueError: If weights length doesn't match number of assets
                or weights don't sum to approximately 1.0
        
        Note:
            Uses numpy.isclose() for floating-point comparison with
            relative tolerance of 1e-5.
        """
        # Validation weights
        if weights is not None:
            if len(weights) != len(self.assets):
                raise ValueError("Weights length doesn't match number of assets")
            if not np.isclose(sum(weights), 1.0, rtol=1e-5):
                raise ValueError("Weights should sum to 1")
        else:
            weights = np.ones(len(self.assets)) / len(self.assets)
        return weights

    def _compute_portfolio_returns(self):
        """
        Calculate portfolio returns using buy-and-hold strategy.
        
        Private method that simulates realistic investment by purchasing assets
        at initial target weights and holding without rebalancing. Share quantities
        remain fixed while portfolio weights drift naturally with price changes.
        
        Process:
            1. Align price data across assets (common trading dates)
            2. Calculate fixed share allocation: shares_i = (weight_i x initial_value) / price_i,0
            3. Track portfolio value: Σ(shares_i x price_i,t)
            4. Compute log returns from portfolio value changes
        
        Returns:
            pd.Series: Daily portfolio log returns with datetime index
        
        Note:
            Portfolio weights drift over time as asset prices evolve differently.
        """
        # Get aligned price data for all assets
        prices_df = pd.concat([asset.prices for asset in self.assets], axis=1).dropna()
        prices_df.columns = [asset.name for asset in self.assets]
        
        # Calculate initial allocation and shares
        initial_prices = prices_df.iloc[0]  # Series with asset prices
        initial_allocation = pd.Series(self.weights, index=initial_prices.index) * self.initial_value  # Series
        shares = initial_allocation / initial_prices  
        
        # Calculate portfolio value over time
        portfolio_values = (prices_df * shares).sum(axis=1)
        
        # Calculate simple returns
        portfolio_simple_returns = portfolio_values.pct_change().dropna()
        
        # Convert to log returns
        log_portfolio_returns = np.log1p(portfolio_simple_returns)
        
        return log_portfolio_returns


    def compute_statistics(self, risk_free_rate=0.02):
        """
        Calculate comprehensive portfolio performance statistics.
        
        Computes portfolio-level metrics including final value, returns,
        volatility, risk-adjusted performance, and maximum drawdown.
        
        Args:
            risk_free_rate (float, optional): Annual risk-free rate.
                Defaults to 0.02 (2%).
        
        Returns:
            dict: Dictionary containing:
                - 'final_value' (float): Portfolio value at end of period
                - 'total_return' (float): Total cumulative return
                - 'annual_return' (float): Annualized return
                - 'annual_volatility' (float): Annualized volatility
                - 'sharpe_ratio' (float): Risk-adjusted return
                - 'risk_free_rate' (float): Input risk-free rate
                - 'max_drawdown' (float): Maximum portfolio decline
        """
        final_value = self.initial_value * self.log_port_returns.cumsum().apply(np.exp).iloc[-1] 
        total_return = np.exp(self.log_port_returns.sum()) - 1
        annual_return = np.exp(self.log_port_returns.mean() * self.trading_days) - 1
        annual_vol = self.log_port_returns.std() * np.sqrt(self.trading_days)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else np.nan

        # Max drawdown
        cum_returns = self.log_port_returns.cumsum().apply(np.exp) 
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        return {
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'risk_free_rate': risk_free_rate,
            'max_drawdown': max_drawdown
        }

    ## === VaR Methods ===

    def historical_var(self):
        """
        Calculate Value at Risk using historical simulation method.
        
        Uses empirical distribution of historical returns to estimate VaR
        without assuming any specific distribution. Scales result for
        specified time horizon.
        
        Returns:
            float: VaR estimate (positive value representing potential loss)
        
        Note:
            - Uses percentile method based on historical return distribution
            - Scales single-period VaR using square-root-of-time rule
        """
        return -np.percentile(self.log_port_returns,100-self.conf_level) * np.sqrt(self.time_horizon)

    def variance_covariance_var(self):
        """
        Calculate parametric VaR using variance-covariance method.
        
        Implements delta-normal approach assuming multivariate normal returns.
        Also computes component VaR showing each asset's contribution to
        portfolio risk.
        
        Returns:
            tuple: (var, component_var) where:
                - var (float): Portfolio VaR estimate
                - component_var (np.array): Each asset's VaR contribution
        
        Note:
            - Assumes returns follow multivariate normal distribution
            - Uses full covariance matrix for correlation effects
            - Component VaR values sum to total portfolio VaR
        """

        returns_df = pd.concat([np.log1p(asset.simple_returns) for asset in self.assets], axis=1)
        returns_df.columns = [asset.name for asset in self.assets]
        cov_matrix = returns_df.cov().values

        portfolio_variance_daily = np.dot(self.weights, np.dot(cov_matrix, self.weights))
        portfolio_sigma_daily = np.sqrt(portfolio_variance_daily)

        sigma_h = portfolio_sigma_daily * np.sqrt(self.time_horizon)
        mu_h = self.mu * self.time_horizon

        Z = norm.ppf((100 - self.conf_level) / 100)
        var = -(mu_h + Z * sigma_h)
        
        marginal_contrib_daily = np.dot(cov_matrix, self.weights) / portfolio_sigma_daily
        component_var = self.weights * marginal_contrib_daily * (-Z) * np.sqrt(self.time_horizon)

        return var, component_var

    def _cornish_fisher_quantile(self, z, skewness, excess_kurtosis):
        """
        Apply Cornish-Fisher expansion to adjust normal quantile.
        
        Private method that corrects normal distribution quantiles for
        skewness and kurtosis using third-order Cornish-Fisher expansion.
        
        Args:
            z (float): Standard normal quantile
            skewness (float): Sample skewness of returns
            excess_kurtosis (float): Sample excess kurtosis (Fisher definition)
        
        Returns:
            float: Adjusted quantile accounting for higher moments
        
        Note:
            Uses third-order expansion including skewness and kurtosis
            corrections for improved tail estimation.
        """
        return (z
            + (1/6)*(z**2 - 1)*skewness
            + (1/24)*(z**3 - 3*z)*excess_kurtosis
            - (1/36)*(2*z**3 - 5*z)*(skewness**2)
        )

    def cornish_fisher_var(self):
        """
        Calculate VaR using Cornish-Fisher expansion method.
        
        Extends parametric VaR by adjusting for skewness and kurtosis in
        return distribution. Provides better tail estimation than standard
        normal assumption.
        
        Returns:
            float: VaR estimate adjusted for higher moments
        
        Note:
            - Computes sample skewness and kurtosis from return history
            - Uses Fisher definition for excess kurtosis (kurtosis - 3)
            - More accurate than normal VaR for non-normal distributions
        """
        skewness = skew(self.log_port_returns)
        excess_kurtosis = kurtosis(self.log_port_returns, fisher=True)  # excess kurtosis = kurtosis - 3
        
        # 3. Quantile standard normal
        z = norm.ppf((100 - self.conf_level)/100)

        # 4. Ajustement de Cornish-Fisher
        z_cf = self._cornish_fisher_quantile(z, skewness, excess_kurtosis)

        # 5. Ajustement pour horizon temporel (ex: 10 jours)
        mu_h = self.mu * self.time_horizon
        sigma_h = self.sigma * np.sqrt(self.time_horizon)

        # 6. VaR ajustée
        var_cf = -(mu_h + z_cf * sigma_h)

        return var_cf
    
    def compute_ewma_variance(self, decay_factor: float = 0.94):
        """
        Calculate Exponentially Weighted Moving Average (EWMA) variance.
        
        Implements EWMA volatility model that gives more
        weight to recent observations. Useful for time-varying volatility.
        
        Args:
            decay_factor (float, optional): Decay parameter (0 < λ < 1).
                Defaults to 0.94 (RiskMetrics standard).
        
        Returns:
            tuple: (current_variance, variance_series) where:
                - current_variance (float): Most recent variance estimate
                - variance_series (list): Historical variance estimates
        
        Note:
            - Higher decay factor gives more persistence in volatility
            - Burn-in period is 3 times the half-life for initialization
        """
        half_life = np.log(0.5) / np.log(decay_factor)
        burn_in = 3 * int(half_life)
        ewma_var = self.log_port_returns.iloc[:burn_in].var(ddof=0)
        variance_over_time = [ewma_var]

        for t in range(burn_in, len(self.log_port_returns)):
            ewma_var = (ewma_var * decay_factor) + ((1 - decay_factor) * (self.log_port_returns.iloc[t] ** 2))
            variance_over_time.append(ewma_var)

        return variance_over_time[-1], variance_over_time

    def ewma_var(self, decay_factor: float = 0.94):
        """
        Calculate VaR using EWMA volatility estimate.
        
        Combines EWMA volatility modeling with normal distribution assumption
        to produce time-varying VaR estimates that adapt to changing market
        conditions.
        
        Args:
            decay_factor (float, optional): EWMA decay parameter.
                Defaults to 0.94 (RiskMetrics standard).
        
        Returns:
            float: VaR estimate using EWMA volatility
        """

        ewma_sigma =  np.sqrt(self.compute_ewma_variance(decay_factor)[0])
        sigma_h = ewma_sigma * np.sqrt(self.time_horizon)
        mu_h = self.mu * self.time_horizon

        Z = norm.ppf((100 - self.conf_level) / 100)
        var = -(mu_h + Z * sigma_h)

        return var
    
    def compute_ewma_var_series(self, decay_factor):
        """
        Calculate time series of EWMA-based VaR estimates.
        
        Produces historical series of VaR estimates using EWMA volatility
        for each time period. Useful for backtesting and time-varying
        risk analysis.
        
        Args:
            decay_factor (float): EWMA decay parameter
        
        Returns:
            np.array: Time series of VaR estimates
        
        Note:
            Length matches the variance series from compute_ewma_variance()
            after burn-in period.
        """
        ewma_sigma =  np.sqrt(self.compute_ewma_variance(decay_factor)[1])
        sigma_h = ewma_sigma * np.sqrt(self.time_horizon)
        mu_h = self.mu * self.time_horizon

        Z = norm.ppf((100 - self.conf_level) / 100)
        var_series = -(mu_h + Z * sigma_h)

        return var_series

    def compute_cvar(self, method: str = "historical"):
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Computes the expected loss given that loss exceeds VaR threshold.
        Provides more information about tail risk than VaR alone.
        
        Args:
            method (str, optional): Calculation method. Options:
                - "historical": Uses empirical distribution
                - "parametric": Assumes normal distribution
                - "cornish-fisher": Adjusts for skewness/kurtosis
                - "EWMA": Uses EWMA volatility estimate
                Defaults to "historical".
        
        Returns:
            float: CVaR estimate
        
        Raises:
            ValueError: If method is not one of the supported options
        """
        alpha = (100 - self.conf_level) / 100

        if method == 'historical':
            var = self.historical_var()
            cvar = -self.log_port_returns[self.log_port_returns <= -var].mean() * np.sqrt(self.time_horizon)

        elif method == 'parametric':
            z = norm.ppf(alpha)
            sigma_h = self.sigma * np.sqrt(self.time_horizon)
            mu_h = self.mu * self.time_horizon
            cvar = mu_h + sigma_h * norm.pdf(z) / alpha
        
        elif method == 'cornish-fisher':
            z = norm.ppf(alpha)
            skewness = skew(self.log_port_returns)
            excess_kurtosis = kurtosis(self.log_port_returns, fisher=True)  # excess kurtosis (Fisher)
            
            # Cornish-Fisher adjusted quantile (third-order)
            z_cf = self._cornish_fisher_quantile(z, skewness, excess_kurtosis)

            sigma_h = self.sigma * np.sqrt(self.time_horizon)
            mu_h = self.mu * self.time_horizon

            # CVaR formula using Cornish-Fisher approximation
            cvar = mu_h + sigma_h * norm.pdf(z_cf) / alpha
        
        elif method == 'EWMA':
            z = norm.ppf(alpha)
            sigma_h = np.sqrt(self.compute_ewma_variance()[0]) * np.sqrt(self.time_horizon)
            mu_h = self.mu * self.time_horizon

            cvar = mu_h + sigma_h * norm.pdf(z) / alpha

        else:
            raise ValueError("Method must be one of: 'historical', 'parametric', 'cornish-fisher', 'EWMA")
        
        return cvar 

    ## === Visualization ===

    def plot_vars(self, decay_factor):
        """
        Create comparative visualization of different VaR methods.
        
        Displays histogram of portfolio returns with VaR estimates from
        all implemented methods overlaid as vertical lines for comparison.
        
        Args:
            decay_factor (float): Decay parameter for EWMA VaR calculation
        
        Returns:
            None: Displays plot using matplotlib
        
        Note:
            Shows historical, parametric, Cornish-Fisher, and EWMA VaR
            estimates on the same chart for direct comparison.
        """
        plt.figure(figsize=(12, 7))
        
        plt.hist(self.log_port_returns, bins=50, alpha=0.7, density=True)
        
        # Different VaR lines
        hist_var = -self.historical_var()
        param_var = -self.variance_covariance_var()[0]
        corfis_var = -self.cornish_fisher_var()
        ewma_var = -self.ewma_var(decay_factor)


        plt.axvline(hist_var, color='red', linestyle='dashed', linewidth=2, label=f'Historical VaR: {hist_var:,.2%}')
        plt.axvline(param_var, color='green', linestyle='dashed', linewidth=2, label=f'Parametric VaR: {param_var:,.2%}')
        plt.axvline(corfis_var, color='purple', linestyle='dashed', linewidth=2, label=f'Cornish-Fisher VaR: {corfis_var:,.2%}')
        plt.axvline(ewma_var, color='blue', linestyle='dashed', linewidth=2, label=f'EWMA VaR: {ewma_var:,.2%}')


        plt.gca().xaxis.set_major_formatter(PercentFormatter(1)) 
        plt.title('Comparison of VaR calculation methods')
        plt.xlabel('Daily gain/loss')
        plt.ylabel('Probability density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_returns(self):
        """
        Display histogram of portfolio returns distribution.
        
        Creates a histogram showing the frequency distribution of daily
        portfolio log returns with percentage formatting on axes.
        
        Returns:
            None: Displays plot using matplotlib
        """
        plt.hist(self.log_port_returns, bins=100, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1)) 
        plt.xlabel('Daily gain/loss')
        plt.ylabel('Probability density')
        plt.title(f"{self.name} Returns histogram")
        plt.show()

    def plot_portfolio_value(self):
        """
        Display time series plot of portfolio value evolution.
        
        Shows how portfolio value changes over time based on cumulative
        returns from the initial investment amount.
        
        Returns:
            None: Displays plot using matplotlib and seaborn
        
        Note:
            Uses log returns to compute cumulative portfolio performance
            and scales by initial portfolio value.
        """
        cumulative_returns = np.exp(self.log_port_returns.cumsum())
        portfolio_value_series = self.initial_value * cumulative_returns

        plt.figure(figsize=(10, 5))
        sns.lineplot(x=portfolio_value_series.index, y=portfolio_value_series, color='navy')
        plt.title("Portfolio value over time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (€)")
        plt.grid(True)
        plt.tight_layout()
        plt.ticklabel_format(style='plain', axis='y')
        plt.show()


class VaRBacktest:
    """
    Simple rolling window backtesting for Value at Risk models.
    
    For each day t, calculates VaR using data from [t-window_size, t-1]
    and tests against the actual loss on day t.
    """
    
    def __init__(self, portfolio: Portfolio, window_size: int = 252):
        """
        Initialize the backtest framework.
        
        Args:
            portfolio (Portfolio): Portfolio object to backtest
            window_size (int): Rolling window size in days (default: 252 = 1 year)
        """
        self.portfolio = portfolio
        self.window_size = window_size
        self.returns = portfolio.log_port_returns
        self.alpha = (100 - portfolio.conf_level) / 100
        
        if len(self.returns) <= window_size:
            raise ValueError(f"Need at least {window_size + 1} observations")
    
    def backtest_historical(self):
        """
        Backtest historical VaR method.
        
        Returns:
            tuple: (violation_rate, violations_list, var_estimates)
        """
        violations = []
        var_estimates = []
        
        for t in range(self.window_size, len(self.returns)):
            # Window data: [t-window_size, t-1]
            window_data = self.returns.iloc[t-self.window_size:t]
            
            # Calculate historical VaR
            var_estimate = -np.percentile(window_data, 100 - self.portfolio.conf_level)
            var_estimates.append(var_estimate)
            
            # Actual loss on day t (scaled for time horizon)
            actual_loss = -self.returns.iloc[t] * np.sqrt(self.portfolio.time_horizon)
            var_scaled = var_estimate * np.sqrt(self.portfolio.time_horizon)
            
            # Check violation
            violation = 1 if actual_loss > var_scaled else 0
            violations.append(violation)
        
        violation_rate = np.mean(violations)
        return violation_rate, violations, var_estimates
    
    def backtest_parametric(self):
        """
        Backtest parametric (variance-covariance) VaR method.
        
        Returns:
            tuple: (violation_rate, violations_list, var_estimates)
        """
        violations = []
        var_estimates = []
        z_score = norm.ppf(self.alpha)
        
        for t in range(self.window_size, len(self.returns)):
            # Window data: [t-window_size, t-1]
            window_data = self.returns.iloc[t-self.window_size:t]
            
            # Calculate parametric VaR
            mu = window_data.mean()
            sigma = window_data.std()
            
            # VaR for time horizon
            mu_h = mu * self.portfolio.time_horizon
            sigma_h = sigma * np.sqrt(self.portfolio.time_horizon)
            var_estimate = -(mu_h + z_score * sigma_h)
            var_estimates.append(var_estimate)
            
            # Actual loss on day t (scaled for time horizon)
            actual_loss = -self.returns.iloc[t] * np.sqrt(self.portfolio.time_horizon)
            
            # Check violation
            violation = 1 if actual_loss > var_estimate else 0
            violations.append(violation)
        
        violation_rate = np.mean(violations)
        return violation_rate, violations, var_estimates
    
    def backtest_cornish_fisher(self):
        """
        Backtest Cornish-Fisher VaR method.
        
        Returns:
            tuple: (violation_rate, violations_list, var_estimates)
        """
        violations = []
        var_estimates = []
        z_score = norm.ppf(self.alpha)
        
        for t in range(self.window_size, len(self.returns)):
            # Window data: [t-window_size, t-1]
            window_data = self.returns.iloc[t-self.window_size:t]
            
            # Calculate moments
            mu = window_data.mean()
            sigma = window_data.std()
            skewness = skew(window_data)
            excess_kurtosis = kurtosis(window_data, fisher=True)
            
            # Cornish-Fisher adjustment
            z_cf = self._cornish_fisher_quantile(z_score, skewness, excess_kurtosis)
            
            # VaR for time horizon
            mu_h = mu * self.portfolio.time_horizon
            sigma_h = sigma * np.sqrt(self.portfolio.time_horizon)
            var_estimate = -(mu_h + z_cf * sigma_h)
            var_estimates.append(var_estimate)
            
            # Actual loss on day t (scaled for time horizon)
            actual_loss = -self.returns.iloc[t] * np.sqrt(self.portfolio.time_horizon)
            
            # Check violation
            violation = 1 if actual_loss > var_estimate else 0
            violations.append(violation)
        
        violation_rate = np.mean(violations)
        return violation_rate, violations, var_estimates
    
    def backtest_ewma(self, decay_factor: float = 0.94):
        """
        Backtest EWMA VaR method.
        
        Args:
            decay_factor (float): EWMA decay factor (default: 0.94)
        
        Returns:
            tuple: (violation_rate, violations_list, var_estimates)
        """
        violations = []
        var_estimates = []
        z_score = norm.ppf(self.alpha)
        
        for t in range(self.window_size, len(self.returns)):
            # Window data: [t-window_size, t-1]
            window_data = self.returns.iloc[t-self.window_size:t]
            
            # Calculate EWMA variance
            ewma_variance = self._compute_ewma_variance(window_data, decay_factor)
            ewma_sigma = np.sqrt(ewma_variance)
            
            # VaR for time horizon
            mu_h = window_data.mean() * self.portfolio.time_horizon
            sigma_h = ewma_sigma * np.sqrt(self.portfolio.time_horizon)
            var_estimate = -(mu_h + z_score * sigma_h)
            var_estimates.append(var_estimate)
            
            # Actual loss on day t (scaled for time horizon)
            actual_loss = -self.returns.iloc[t] * np.sqrt(self.portfolio.time_horizon)
            
            # Check violation
            violation = 1 if actual_loss > var_estimate else 0
            violations.append(violation)
        
        violation_rate = np.mean(violations)
        return violation_rate, violations, var_estimates
    
    def _cornish_fisher_quantile(self, z, skewness, excess_kurtosis):
        """
        Apply Cornish-Fisher expansion to adjust normal quantile.
        
        Private method that corrects normal distribution quantiles for
        skewness and kurtosis using third-order Cornish-Fisher expansion.
        
        Args:
            z (float): Standard normal quantile
            skewness (float): Sample skewness of returns
            excess_kurtosis (float): Sample excess kurtosis (Fisher definition)
        
        Returns:
            float: Adjusted quantile accounting for higher moments
        
        Note:
            Uses third-order expansion including skewness and kurtosis
            corrections for improved tail estimation.
        """
        return (z
            + (1/6)*(z**2 - 1)*skewness
            + (1/24)*(z**3 - 3*z)*excess_kurtosis
            - (1/36)*(2*z**3 - 5*z)*(skewness**2)
        )
    
    def _compute_ewma_variance(self, returns: pd.Series, decay_factor: float = 0.94):
        """
        Calculate Exponentially Weighted Moving Average (EWMA) variance.
        
        Implements EWMA volatility model that gives more
        weight to recent observations. Useful for time-varying volatility.
        
        Args:
            decay_factor (float, optional): Decay parameter (0 < λ < 1).
                Defaults to 0.94 (RiskMetrics standard).
        
        Returns:
            tuple: (current_variance, variance_series) where:
                - current_variance (float): Most recent variance estimate
                - variance_series (list): Historical variance estimates
        
        Note:
            - Higher decay factor gives more persistence in volatility
            - Burn-in period is 3 times the half-life for initialization
        """
        half_life = np.log(0.5) / np.log(decay_factor)
        burn_in = 3 * int(half_life)
        ewma_var = returns.iloc[:burn_in].var(ddof=0)

        for t in range(burn_in, len(returns)):
            ewma_var = (ewma_var * decay_factor) + ((1 - decay_factor) * (returns.iloc[t] ** 2))

        return ewma_var
    
    def run_all_backtests(self, ewma_decay: float = 0.94):
        """
        Run all backtesting methods and return results.
        
        Args:
            ewma_decay (float): EWMA decay factor (default: 0.94)
        
        Returns:
            dict: Results for all methods
        """
        print(f"Running backtests with {self.window_size}-day rolling window")
        print(f"Total test observations: {len(self.returns) - self.window_size}")
        print(f"Expected violation rate: {self.alpha:.2%}")
        print("-" * 50)
        
        results = {}
        
        # Historical method
        violation_rate, violations, var_est = self.backtest_historical()
        results['historical'] = {
            'violation_rate': violation_rate,
            'violations': violations,
            'var_estimates': var_est
        }
        print(f"Historical VaR:      {violation_rate:.2%} violation rate")
        
        # Parametric method
        violation_rate, violations, var_est = self.backtest_parametric()
        results['parametric'] = {
            'violation_rate': violation_rate,
            'violations': violations,
            'var_estimates': var_est
        }
        print(f"Parametric VaR:      {violation_rate:.2%} violation rate")
        
        # Cornish-Fisher method
        violation_rate, violations, var_est = self.backtest_cornish_fisher()
        results['cornish_fisher'] = {
            'violation_rate': violation_rate,
            'violations': violations,
            'var_estimates': var_est
        }
        print(f"Cornish-Fisher VaR:  {violation_rate:.2%} violation rate")
        
        # EWMA method
        violation_rate, violations, var_est = self.backtest_ewma(ewma_decay)
        results['ewma'] = {
            'violation_rate': violation_rate,
            'violations': violations,
            'var_estimates': var_est,
            'decay_factor': ewma_decay
        }
        print(f"EWMA VaR (λ={ewma_decay}):   {violation_rate:.2%} violation rate")
        
        return results
