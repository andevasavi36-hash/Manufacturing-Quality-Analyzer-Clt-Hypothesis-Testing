import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class ManufacturingQualityAnalyzer:
    """
    Manufacturing Quality Analyzer
    --------------------------------
    Uses:
    - Central Limit Theorem (CLT)
    - Hypothesis Testing
    - p-value estimation
    - Confidence Intervals
    - Control Limits
    """

    def __init__(self, target_mean, sigma=None, alpha=0.05):
        """
        Parameters:
        -----------
        target_mean : float
            Expected production mean.

        sigma : float
            Known population standard deviation.
            If None, sample standard deviation is used.

        alpha : float
            Significance level.
        """

        self.target_mean = target_mean
        self.sigma = sigma
        self.alpha = alpha

    # --------------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------------
    def summarize_data(self, data):
        data = np.array(data)

        summary = {
            "Sample Size": len(data),
            "Sample Mean": np.mean(data),
            "Sample Std": np.std(data, ddof=1),
            "Minimum": np.min(data),
            "Maximum": np.max(data),
        }

        return pd.DataFrame(summary.items(), columns=["Metric", "Value"])

    # --------------------------------------------------------
    # CENTRAL LIMIT THEOREM ANALYSIS
    # --------------------------------------------------------
    def clt_analysis(self, data, num_samples=1000, sample_size=30):
        """
        Generate sampling distribution using CLT.
        """

        data = np.array(data)
        sample_means = []

        for _ in range(num_samples):
            sample = np.random.choice(data, size=sample_size, replace=True)
            sample_means.append(np.mean(sample))

        sample_means = np.array(sample_means)

        print("\n--- CENTRAL LIMIT THEOREM ANALYSIS ---")
        print(f"Original Mean      : {np.mean(data):.4f}")
        print(f"Sampling Mean      : {np.mean(sample_means):.4f}")
        print(f"Sampling Std Error : {np.std(sample_means):.4f}")

        # Plot sampling distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sample_means, bins=30, density=True)
        plt.title("Sampling Distribution of the Mean (CLT)")
        plt.xlabel("Sample Mean")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

        return sample_means

    # --------------------------------------------------------
    # ONE-SAMPLE HYPOTHESIS TEST
    # --------------------------------------------------------
    def hypothesis_test(self, data, alternative='two-sided'):
        """
        Perform z-test or t-test.

        H0: mean == target_mean
        H1: mean != target_mean
        """

        data = np.array(data)

        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)

        print("\n--- HYPOTHESIS TEST ---")
        print(f"Null Hypothesis Mean (H0): {self.target_mean}")
        print(f"Sample Mean             : {sample_mean:.4f}")
        print(f"Sample Size             : {n}")

        # Z-Test when sigma known
        if self.sigma is not None:
            standard_error = self.sigma / np.sqrt(n)
            z_stat = (sample_mean - self.target_mean) / standard_error

            if alternative == 'two-sided':
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            elif alternative == 'greater':
                p_value = 1 - stats.norm.cdf(z_stat)
            else:
                p_value = stats.norm.cdf(z_stat)

            print("Test Type : Z-Test")
            print(f"Z Statistic : {z_stat:.4f}")

            test_statistic = z_stat

        # T-Test when sigma unknown
        else:
            t_stat, p_value = stats.ttest_1samp(data, self.target_mean)
            print("Test Type : T-Test")
            print(f"T Statistic : {t_stat:.4f}")

            test_statistic = t_stat

        print(f"P-Value : {p_value:.6f}")

        if p_value < self.alpha:
            decision = "Reject Null Hypothesis"
        else:
            decision = "Fail to Reject Null Hypothesis"

        print(f"Decision : {decision}")

        return {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "decision": decision
        }

    # --------------------------------------------------------
    # CONFIDENCE INTERVAL
    # --------------------------------------------------------
    def confidence_interval(self, data, confidence=0.95):
        data = np.array(data)

        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        standard_error = std / np.sqrt(n)

        margin = stats.t.ppf((1 + confidence) / 2, df=n - 1) * standard_error

        lower = mean - margin
        upper = mean + margin

        print("\n--- CONFIDENCE INTERVAL ---")
        print(f"Confidence Level : {confidence * 100:.1f}%")
        print(f"Mean             : {mean:.4f}")
        print(f"Interval          : ({lower:.4f}, {upper:.4f})")

        return lower, upper

    # --------------------------------------------------------
    # PROCESS CAPABILITY
    # --------------------------------------------------------
    def process_capability(self, data, lsl, usl):
        """
        Calculate Cp and Cpk.
        """

        data = np.array(data)

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        cp = (usl - lsl) / (6 * std)

        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)

        cpk = min(cpu, cpl)

        print("\n--- PROCESS CAPABILITY ---")
        print(f"Cp  : {cp:.4f}")
        print(f"Cpk : {cpk:.4f}")

        if cpk >= 1.33:
            print("Process is capable.")
        else:
            print("Process needs improvement.")

        return {
            "Cp": cp,
            "Cpk": cpk
        }

    # --------------------------------------------------------
    # CONTROL CHART
    # --------------------------------------------------------
    def control_chart(self, data):
        """
        Create simple X-bar control chart.
        """

        data = np.array(data)

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        ucl = mean + 3 * std
        lcl = mean - 3 * std

        plt.figure(figsize=(12, 6))
        plt.plot(data, marker='o')
        plt.axhline(mean, linestyle='--', label='Mean')
        plt.axhline(ucl, linestyle='--', label='UCL (+3σ)')
        plt.axhline(lcl, linestyle='--', label='LCL (-3σ)')

        plt.title('Manufacturing Quality Control Chart')
        plt.xlabel('Sample Number')
        plt.ylabel('Measurement')
        plt.legend()
        plt.grid(True)
        plt.show()


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":

    # Simulated manufacturing measurements
    np.random.seed(42)

    production_data = np.random.normal(
        loc=50,      # target mean
        scale=2,     # process std deviation
        size=200
    )

    analyzer = ManufacturingQualityAnalyzer(
        target_mean=50,
        sigma=2,
        alpha=0.05
    )

    # 1. Data Summary
    summary = analyzer.summarize_data(production_data)
    print(summary)

    # 2. CLT Analysis
    analyzer.clt_analysis(
        production_data,
        num_samples=1000,
        sample_size=30
    )

    # 3. Hypothesis Test
    analyzer.hypothesis_test(production_data)

    # 4. Confidence Interval
    analyzer.confidence_interval(production_data)

    # 5. Process Capability
    analyzer.process_capability(
        production_data,
        lsl=45,
        usl=55
    )

    # 6. Control Chart
    analyzer.control_chart(production_data)
