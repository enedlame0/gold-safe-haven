import pandas as pd
import numpy as np
from arch.__future__ import reindexing
from arch.univariate import ConstantMean, GARCH
from arch.multivariate import ConstantCorrelation, DynamicConditionalCorrelation
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
    

# --- Your Setup Code ---
df = pd.read_csv(r"C:\Users\Ened Lame\Desktop\business project\data.csv")

# Convert to proper date time format
df["Date"] = pd.to_datetime(df["Date"], unit="D", origin="1899-12-30")

# Calculate daily log returns
df["R_XAU"] = np.log(df["XAU"] / df["XAU"].shift(1))
df["R_SPX"] = np.log(df["SPX"] / df["SPX"].shift(1))
df["R_USD"] = np.log(df["LUATTRUU"] / df["LUATTRUU"].shift(1))

# Rename for clarity and drop NaNs
df.rename(columns={"R_USD": "R_Bond"}, inplace=True)
df = df.dropna()

# Limit the time period to 1 Mar 2007 to 1 Apr 2009
# df = df[(df["Date"] >= "2007-03-01") & (df["Date"] <= "2009-04-01")].copy()

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["R_XAU"], label="Gold Returns (R_XAU)", color="gold")
plt.plot(df["Date"], df["R_SPX"], label="Market Returns (R_SPX)", color="blue")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.title("Daily Log Returns: Gold (XAU) vs. Market (SPX)")
plt.legend()
plt.show()

def run_mgarch_dcc_analysis(data, ret1, ret2, start_date, end_date):
    """
    Runs a MGARCH DCC(1,1) analysis on two return series between start_date and end_date.
    Plots the dynamic conditional correlation.
    """
    # Subset and clean data
    df_sub = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)].copy()
    dcc_df = df_sub[[ret1, ret2]].dropna()

    # Fit univariate GARCH(1,1) models for each series
    models = []
    for col in dcc_df.columns:
        am = ConstantMean(dcc_df[col])
        am.volatility = GARCH(1, 0, 1)
        res = am.fit(disp="off")
        models.append(res)

    # Fit DCC(1,1) model
    dcc = DynamicConditionalCorrelation(models)
    dcc_res = dcc.fit()

    print(dcc_res.summary())

    # Plot the dynamic conditional correlation
    plt.figure(figsize=(12, 6))
    corr = dcc_res.corr.loc[:, (ret1, ret2)]
    plt.plot(corr.index, corr, label=f"{ret1} vs {ret2} DCC")
    plt.title(f"DCC-GARCH(1,1) Dynamic Correlation: {ret1} vs {ret2}\n{start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()
dcc_df = df[["R_XAU", "R_SPX", "R_Bond"]].dropna().copy()


# --- Reusable Function to Run, Print, and Plot Quantile Regression ---
# def run_and_plot_quantile_regression(
#     data, formula, dependent_var, independent_var, title
# ):
#     """
#     Runs a quantile regression, prints a summary table of the coefficients,
#     and plots the results.
#     """
#     print(f"\n--- Running Analysis for: {title} ---")
#     quantiles = [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]
#     results_list = []  # To store results for the summary table

#     # --- Run Quantile Regressions ---
#     for q in quantiles:
#         model = smf.quantreg(formula, data).fit(q=q, vcov="robust")
#         # For each variable in the model, add its results to our list
#         for var in model.params.index:
#             results_list.append(
#                 {
#                     "Model": f"Quantile {q}",
#                     "Variable": var,
#                     "Coefficient (Beta)": model.params[var],
#                     "Std. Error": model.bse[var],
#                     "P-Value": model.pvalues[var],
#                 }
#             )

#     # --- Run OLS with HAC errors for comparison ---
#     ols_model = smf.ols(formula, data).fit(
#         cov_type="HAC", cov_kwds={"maxlags": 1}
#     )
#     for var in ols_model.params.index:
#         results_list.append(
#             {
#                 "Model": "OLS",
#                 "Variable": var,
#                 "Coefficient (Beta)": ols_model.params[var],
#                 "Std. Error": ols_model.bse[var],
#                 "P-Value": ols_model.pvalues[var],
#             }
#         )

#     # --- Create and Print the Summary Table ---
#     summary_df = pd.DataFrame(results_list)
#     print("\nRegression Results Summary:")
#     # Use set_index for a cleaner, multi-level table format
#     print(
#         summary_df.set_index(["Model", "Variable"]).round(4).to_string()
#     )

#     # --- Prepare data for plotting (from the DataFrame we just made) ---
#     plot_data = summary_df[
#         (summary_df["Variable"] == independent_var)
#         & (summary_df["Model"] != "OLS")
#     ].copy()
#     plot_data["Quantile"] = plot_data["Model"].str.split().str[1].astype(float)
#     plot_data.sort_values("Quantile", inplace=True)

#     # Calculate confidence intervals for the plot
#     plot_data["lower_ci"] = (
#         plot_data["Coefficient (Beta)"] - 1.96 * plot_data["Std. Error"]
#     )
#     plot_data["upper_ci"] = (
#         plot_data["Coefficient (Beta)"] + 1.96 * plot_data["Std. Error"]
#     )

#     ols_beta = ols_model.params[independent_var]
#     ols_ci = ols_model.conf_int().loc[independent_var]

#     # --- Create the plot ---
#     plt.style.use("seaborn-v0_8-whitegrid")
#     fig, ax = plt.subplots(figsize=(10, 6))

#     ax.plot(
#         plot_data["Quantile"],
#         plot_data["Coefficient (Beta)"],
#         "b-o",
#         label=f"Quantile β for {independent_var}",
#     )
#     ax.fill_between(
#         plot_data["Quantile"],
#         plot_data["lower_ci"],
#         plot_data["upper_ci"],
#         alpha=0.2,
#         color="blue",
#         label="95% CI",
#     )
#     ax.axhline(
#         ols_beta,
#         color="red",
#         linestyle="--",
#         label=f"OLS β ({ols_beta:.2f})",
#     )
#     ax.axhline(
#         ols_ci[0], color="red", linestyle=":", alpha=0.5, label="OLS 95% CI"
#     )
#     ax.axhline(ols_ci[1], color="red", linestyle=":", alpha=0.5)
#     ax.axhline(0, color="black", linestyle="-", linewidth=1)

#     ax.set_xlabel(f"Quantile (τ) of {dependent_var} Conditional Distribution")
#     ax.set_ylabel(f"Market Beta (β) for {independent_var} Returns")
#     ax.set_title(title)
#     ax.legend()
#     plt.show()


def run_and_plot_quantile_regression(
    data, formula, dependent_var, independent_var, title, quantiles = None, vcov_type = "robust", vcv_kwds = None
):
    """
    Runs a quantile regression, prints a summary table of the coefficients,
    and plots the results.
    """
    print(f"\n--- Running Analysis for: {title} ---")
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.10, 0.20]
    if vcv_kwds is None:
        vcv_kwds = {"kernel": "gau", "bandwidth": "hsheather"}
    
    if "- 1" not in formula and "-1" not in formula:
        formula = formula.strip() + " - 1"

    results_list = []

    # --- Run Quantile Regressions ---
    for q in quantiles:
        model = smf.quantreg(formula, data).fit(q=q, vcov=vcov_type, **vcv_kwds)
        # For each variable in the model, add its results to our list
        for var in model.params.index:
            results_list.append(
                {
                    "Model": f"Quantile {q}",
                    "Variable": var,
                    "Coefficient (Beta)": model.params[var],
                    "Std. Error": model.bse[var],
                    "P-Value": model.pvalues[var],
                }
            )

    # --- Run OLS with HAC errors for comparison ---
    ols_model = smf.ols(formula, data).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )
    for var in ols_model.params.index:
        results_list.append(
            {
                "Model": "OLS",
                "Variable": var,
                "Coefficient (Beta)": ols_model.params[var],
                "Std. Error": ols_model.bse[var],
                "P-Value": ols_model.pvalues[var],
            }
        )

    # --- Create and Print the Summary Table ---
    summary_df = pd.DataFrame(results_list)
    print("\nRegression Results Summary:")
    # Use set_index for a cleaner, multi-level table format
    print(
        summary_df.set_index(["Model", "Variable"]).round(4).to_string()
    )

    # --- Prepare data for plotting (from the DataFrame we just made) ---
    plot_data = summary_df[
        (summary_df["Variable"] == independent_var)
        & (summary_df["Model"] != "OLS")
    ].copy()
    plot_data["Quantile"] = plot_data["Model"].str.split().str[1].astype(float)
    plot_data.sort_values("Quantile", inplace=True)

    # Calculate confidence intervals for the plot
    plot_data["lower_ci"] = (
        plot_data["Coefficient (Beta)"] - 1.96 * plot_data["Std. Error"]
    )
    plot_data["upper_ci"] = (
        plot_data["Coefficient (Beta)"] + 1.96 * plot_data["Std. Error"]
    )

    ols_beta = ols_model.params[independent_var]
    ols_ci = ols_model.conf_int().loc[independent_var]

    # --- Create the plot ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        plot_data["Quantile"],
        plot_data["Coefficient (Beta)"],
        "b-o",
        label=f"Quantile β for {independent_var}",
    )
    ax.fill_between(
        plot_data["Quantile"],
        plot_data["lower_ci"],
        plot_data["upper_ci"],
        alpha=0.2,
        color="blue",
        label="95% CI",
    )
    ax.axhline(
        ols_beta,
        color="red",
        linestyle="--",
        label=f"OLS β ({ols_beta:.2f})",
    )
    ax.axhline(
        ols_ci[0], color="red", linestyle=":", alpha=0.5, label="OLS 95% CI"
    )
    ax.axhline(ols_ci[1], color="red", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linestyle="-", linewidth=1)

    ax.set_xlabel(f"Quantile (τ) of {dependent_var} Conditional Distribution")
    ax.set_ylabel(f"Market Beta (β) for {independent_var} Returns")
    ax.set_title(title)
    ax.legend()
    plt.show()

def run_interaction_analysis(data, dependent_var, title):
    """
    Runs a quantile regression with a market distress interaction term
    and plots the interaction coefficient (delta).
    """
    print(f"\n--- Running Interaction Analysis for: {title} ---")

    # Create a copy to avoid modifying the original DataFrame
    df_copy = data.copy()

    # Create the Market Distress indicator variable
    distress_threshold = df_copy["R_SPX"].quantile(0.10)
    df_copy["Market_Distress"] = (
        df_copy["R_SPX"] <= distress_threshold
    ).astype(int)

    # Define the formula with the interaction term
    formula = f"{dependent_var} ~ R_SPX + R_SPX:Market_Distress"
    interaction_term_name = "R_SPX:Market_Distress"
    quantiles = [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]
    results_list = []

    # Run the quantile regression for each quantile
    for q in quantiles:
        model = smf.quantreg(formula, df_copy).fit(q=q, vcov="robust")
        for var in model.params.index:
            results_list.append(
                {
                    "Model": f"Quantile {q}",
                    "Variable": var,
                    # Changed name for clarity
                    "Coefficient": model.params[var],
                    "Std. Error": model.bse[var],
                    "P-Value": model.pvalues[var],
                }
            )

    # Create and print the summary table
    summary_df = pd.DataFrame(results_list)
    print("\nRegression Results Summary (Interaction Model):")
    print(summary_df.set_index(["Model", "Variable"]).round(4).to_string())

    # Isolate the interaction term results for plotting
    plot_data = summary_df[
        summary_df["Variable"] == interaction_term_name
    ].copy()
    plot_data["Quantile"] = plot_data["Model"].str.split().str[1].astype(float)
    plot_data.sort_values("Quantile", inplace=True)

    # Calculate confidence intervals for the plot
    plot_data["lower_ci"] = (
        plot_data["Coefficient"] - 1.96 * plot_data["Std. Error"]
    )
    plot_data["upper_ci"] = (
        plot_data["Coefficient"] + 1.96 * plot_data["Std. Error"]
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        plot_data["Quantile"],
        plot_data["Coefficient"],
        "g-o",
        label="Interaction Effect δ(τ)",
    )
    ax.fill_between(
        plot_data["Quantile"],
        plot_data["lower_ci"],
        plot_data["upper_ci"],
        alpha=0.2,
        color="green",
        label="95% CI for δ(τ)",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Quantile (τ) of {dependent_var} Conditional Distribution")
    ax.set_ylabel("Interaction Effect δ on Market Beta")
    ax.set_title(title)
    ax.legend()
    plt.show()

def run_rolling_correlation_analysis(data, asset1_ret, asset2_ret, window, title, start, end):
    print(f"\n--- Running Rolling Correlation Analysis for: {title} ---")

    df_copy = data.copy()
    if "Date" in df_copy.columns:
        df_copy.set_index("Date", inplace=True)

    rolling_corr = (
        df_copy[asset1_ret]
        .rolling(window=window)
        .corr(df_copy[asset2_ret])
    )

    # Restrict to the specified date range for plotting
    rolling_corr = rolling_corr.loc[start:end]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        rolling_corr.index,
        rolling_corr,
        label=f"{window}-Day Rolling Correlation",
        color="teal",
    )

    # Add a horizontal line at zero for reference
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # Set labels and title
    ax.set_title(title)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    ax.set_ylim(-1, 1)  # Correlation is always between -1 and 1
    ax.legend()
    plt.show()


rolling_window = 250

startd = "2000-01-01"
endd = "2024-12-31"
df_sub = df[(df["Date"] >= startd) & (df["Date"] <= endd)].copy()

run_rolling_correlation_analysis(
    data=df,
    asset1_ret="R_XAU",
    asset2_ret="R_SPX",
    window=rolling_window,
    title=f"{rolling_window}-Day Rolling Correlation: Gold (XAU) vs. Market (SPX)",
    start = startd,
    end = endd,
)

run_rolling_correlation_analysis(
    data=df,
    asset1_ret="R_Bond",
    asset2_ret="R_SPX",
    window=rolling_window,
    title=f"{rolling_window}-Day Rolling Correlation: Bonds vs. Market (SPX)",
    start = startd,
    end = endd,
)

# --- CRISIS-SPECIFIC ANALYSES ---
# Define the localized crisis windows
crisis_windows = {
    "DotCom": ("2000-03-01", "2002-10-01"),
    "GFC":    ("2007-07-01", "2009-03-31"),
    "COVID":  ("2020-01-01", "2020-12-31"),
}

# rolling_window = 250

# for name, (start, end) in crisis_windows.items():
#     # Subset the data for this crisis
#     df_sub = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
#     print(f"\n\n===== {name} Crisis: {start} to {end} =====\n")

#     # 1) Quantile regression: Gold vs. Market
#     run_and_plot_quantile_regression(
#         data=df_sub,
#         formula="R_XAU ~ R_SPX",
#         dependent_var="R_XAU",
#         independent_var="R_SPX",
#         title=f"Gold ∼ Market during {name} Crisis",
#     )

#     # 2) Quantile regression: Bonds vs. Market
#     run_and_plot_quantile_regression(
#         data=df_sub,
#         formula="R_Bond ~ R_SPX",
#         dependent_var="R_Bond",
#         independent_var="R_SPX",
#         title=f"Bonds ∼ Market during {name} Crisis",
#     )

#     # 3) Interaction analysis (safe-haven test)
#     run_interaction_analysis(
#         data=df_sub,
#         dependent_var="R_XAU",
#         title=f"Gold Safe-Haven Interaction: {name} Crisis",
#     )
#     run_interaction_analysis(
#         data=df_sub,
#         dependent_var="R_Bond",
#         title=f"Bond Safe-Haven Interaction: {name} Crisis",
#     )

#     # 4) Rolling correlation
#     run_rolling_correlation_analysis(
#         data=df_sub,
#         asset1_ret="R_XAU",
#         asset2_ret="R_SPX",
#         window=rolling_window,
#         title=(f"{rolling_window}-Day Rolling Corr: XAU vs SPX "
#                f"during {name} Crisis"),
#     )
#     run_rolling_correlation_analysis(
#         data=df_sub,
#         asset1_ret="R_Bond",
#         asset2_ret="R_SPX",
#         window=rolling_window,
#         title=(f"{rolling_window}-Day Rolling Corr: Bond vs SPX "
#                f"during {name} Crisis"),
#     )