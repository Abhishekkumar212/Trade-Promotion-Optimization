from fastapi import FastAPI, Request, Form, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
import uuid
import os
from typing import Dict, Any, Tuple, Optional, List
from starlette.middleware.sessions import SessionMiddleware
# from starlette.requests import Request as StarletteRequest # Not strictly needed if only using FastAPI's Request

plt.style.use('seaborn-v0_8-whitegrid')

app = FastAPI()
# IMPORTANT: Change secret_key to a strong, random string for production
app.add_middleware(SessionMiddleware, secret_key="a-very-secret-key-please-change-me")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

sessions: Dict[str, Dict[str, Any]] = {}

OPTIMIZER_VERBOSE_DEBUG = False # Set to True for detailed optimizer console output

# --- Function Definitions (Moved to the top) ---

def save_plot_to_base64(fig: plt.Figure) -> str:
    """Saves a Matplotlib figure to a base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig) # Close the figure to free memory
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def generate_category_data(category_name: str, num_products: int,
                           base_volume_range: tuple,
                           markup_factor_range_fallback: tuple,
                           discount_uplift_range: tuple,
                           product_name_prefixes: list,
                           real_prices: list = None):
    np.random.seed(hash(category_name) % (2**32 - 1))
    if len(product_name_prefixes) < num_products:
        generic_suffixes = [f"Variety {i+1}" for i in range(num_products - len(product_name_prefixes))]
        product_names = product_name_prefixes + [f"{product_name_prefixes[-1] if product_name_prefixes else 'Product'} {s}" for s in generic_suffixes]
    else:
        product_names = product_name_prefixes[:num_products]

    base_volumes = np.random.randint(base_volume_range[0], base_volume_range[1], size=num_products)
    list_base_price = np.zeros(num_products)
    per_unit_cogs = np.zeros(num_products)

    for i in range(num_products):
        if real_prices and i < len(real_prices) and real_prices[i] is not None and real_prices[i] > 0:
            list_base_price[i] = float(real_prices[i])
            cogs_ratio = np.random.uniform(0.3, 0.7)
            per_unit_cogs[i] = np.round(list_base_price[i] * cogs_ratio, 2)
        else:
            fallback_cogs_value = np.random.uniform(1.0, 50.0)
            per_unit_cogs[i] = np.round(fallback_cogs_value,2)
            markup_factor = np.random.uniform(markup_factor_range_fallback[0], markup_factor_range_fallback[1])
            generated_price = np.round(per_unit_cogs[i] * markup_factor, 2)
            list_base_price[i] = np.maximum(generated_price, np.round(per_unit_cogs[i] * 1.1, 2) + 0.01)
    per_unit_cogs = np.minimum(per_unit_cogs, list_base_price * 0.98)

    units = base_volumes.copy()
    per_unit_selling_price = list_base_price.copy()
    discount_percentage = np.zeros(num_products)
    revenue = units * per_unit_selling_price
    per_unit_margin = per_unit_selling_price - per_unit_cogs
    margin = units * per_unit_margin
    per_unit_margin = np.maximum(0, per_unit_margin)
    margin = np.maximum(0, margin)

    discount_uplift = np.round(np.random.uniform(discount_uplift_range[0], discount_uplift_range[1], size=num_products), 4)
    tactic_uplift = np.ones(num_products)

    data = {
        'PL4 Sub-Category': list(product_names),
        'Forecast Avg. Base Volume': list(base_volumes.astype(int)),
        'Units': list(units.astype(int)),
        'Per Unit COGS': list(np.round(per_unit_cogs, 2)),
        'List/ Base Price': list(np.round(list_base_price,2)),
        'Per Unit Selling Price': list(np.round(per_unit_selling_price,2)),
        'Discount': list(discount_percentage),
        'Revenue': list(np.round(revenue, 2)),
        'Per Unit Margin': list(np.round(per_unit_margin, 2)),
        'Margin': list(np.round(margin, 2)),
        'Per Unit Rebate': list(np.zeros(num_products)),
        'Dollor Investment': list(np.zeros(num_products)),
        'Discount Uplift': list(discount_uplift),
        'Tactic Uplift': list(tactic_uplift.astype(int))
    }
    return data

def format_value(value, kind="currency", currency_symbol="$"):
    if pd.isna(value) or value == '' or value is None: return ''
    try:
        val_float = float(value)
        if kind == "currency": return f"{currency_symbol}{val_float:,.2f}"
        elif kind == "units": return f"{val_float:,.0f}"
        elif kind == "percentage": return f"{val_float:.2f}%"
        elif kind == "general_float": return f"{val_float:,.2f}"
        else: return str(value)
    except (ValueError, TypeError): return str(value)

def create_total_df_structure():
    return pd.DataFrame({
        'Data-Totals': ['Units', 'Revenue', 'Margins', 'Investment'],
        'Planned': [0.0, 0.0, 0.0, 0.0],
        'Optimized': [0.0, 0.0, 0.0, 0.0],
        '% Change': ['', '', '', '']
    })

def create_full_category_total_df_structure():
     return pd.DataFrame({
        'Data-Totals': ['Units', 'Revenue', 'Margins', 'Investment'],
        'Planned': [0.0, 0.0, 0.0, 0.0],
        'Optimized State': [0.0, 0.0, 0.0, 0.0],
        '% Change': ['', '', '', '']
    })

def optimize_discounts(df_input_planned: pd.DataFrame, max_investment: float,
                       lower_discount_pct: float, upper_discount_pct: float,
                       optimization_goal: str = "Revenue") -> Tuple[pd.DataFrame, Optional[str]]:

    df_planned_static_values = df_input_planned.copy()
    planned_margins_per_product = df_planned_static_values['Margin'].values.astype(float)
    planned_margins_per_product = np.nan_to_num(planned_margins_per_product, nan=0.0)
    df_opt = df_input_planned.copy()

    if not (0 <= lower_discount_pct <= 100 and 0 <= upper_discount_pct <= 100 and lower_discount_pct <= upper_discount_pct):
        return pd.DataFrame(), "Input Error: Invalid discount range. Ensure 0 <= Lower <= Upper <= 100."
    if max_investment < 0:
        return pd.DataFrame(), "Input Error: Maximum investment cannot be negative."

    if lower_discount_pct > 1e-6:
        df_temp_check = df_planned_static_values[df_planned_static_values['Forecast Avg. Base Volume'] > 0].copy()
        if not df_temp_check.empty:
            hypothetical_units_at_lower_bound = df_temp_check['Forecast Avg. Base Volume'] * \
                                                np.exp(df_temp_check['Discount Uplift'] * lower_discount_pct / 100) * \
                                                df_temp_check['Tactic Uplift']
            investment_at_lower_bound = (lower_discount_pct / 100) * df_temp_check['List/ Base Price'] * \
                                        hypothetical_units_at_lower_bound
            min_required_investment = investment_at_lower_bound.sum()
            if min_required_investment > max_investment + 1e-3: # Add small tolerance for float comparisons
                return pd.DataFrame(), (
                    f"Configuration Issue: Applying the minimum discount of {lower_discount_pct:.2f}% to all selected products "
                    f"requires an investment of approx. {format_value(min_required_investment)}, "
                    f"which exceeds the maximum allowed investment of {format_value(max_investment)} for this selection. "
                    "Suggestions: Increase Max Investment, or reduce the Lower Discount Range for selected products."
                )

    num_products = len(df_opt)
    if num_products == 0:
        return pd.DataFrame(), "No products selected for optimization."

    x0 = np.full(num_products, np.clip((lower_discount_pct + upper_discount_pct) / 2, lower_discount_pct, upper_discount_pct))
    bounds_list = [(lower_discount_pct, upper_discount_pct)] * num_products
    MARGIN_CONSTRAINT_TOLERANCE = 1e-3 # Allow optimized margin to be slightly less due to float precision

    def calculate_all_metrics_for_optimizer(current_discount_percentages_raw):
        current_discount_percentages = np.clip(current_discount_percentages_raw, lower_discount_pct, upper_discount_pct)
        base_volumes = df_planned_static_values['Forecast Avg. Base Volume'].values
        discount_uplifts = df_planned_static_values['Discount Uplift'].values
        tactic_uplifts = df_planned_static_values['Tactic Uplift'].values
        list_prices = df_planned_static_values['List/ Base Price'].values
        cogs_per_unit = df_planned_static_values['Per Unit COGS'].values

        product_optimized_units = base_volumes * np.exp(discount_uplifts * current_discount_percentages / 100) * tactic_uplifts
        product_absolute_discount_value_per_unit = list_prices * current_discount_percentages / 100
        product_net_selling_price_per_unit = np.maximum(1e-6, list_prices - product_absolute_discount_value_per_unit) # Avoid zero or negative prices

        product_optimized_revenues = product_optimized_units * product_net_selling_price_per_unit
        product_optimized_cogs_total = product_optimized_units * cogs_per_unit
        product_optimized_margins = product_optimized_revenues - product_optimized_cogs_total

        total_revenue = np.sum(product_optimized_revenues)
        total_margin = np.sum(product_optimized_margins)
        total_investment_cost = np.sum(product_absolute_discount_value_per_unit * product_optimized_units)
        return total_revenue, total_investment_cost, total_margin, product_optimized_margins

    def objective_function(current_discount_percentages):
        total_revenue, _, total_margin, _ = calculate_all_metrics_for_optimizer(current_discount_percentages)
        obj_val = 0
        if optimization_goal == "Margin": obj_val = -total_margin
        elif optimization_goal == "Both": obj_val = -(total_revenue + total_margin) # Simple sum, could be weighted
        else: obj_val = -total_revenue # Default to Revenue
        return obj_val if pd.notna(obj_val) else 1e12 # Large penalty for NaN

    def investment_constraint_func(current_discount_percentages):
        _, total_investment_cost, _, _ = calculate_all_metrics_for_optimizer(current_discount_percentages)
        # Constraint: total_investment_cost <= max_investment  =>  max_investment - total_investment_cost >= 0
        return max_investment - (total_investment_cost if pd.notna(total_investment_cost) else max_investment + 1) # Penalize NaN investment

    def product_margin_constraint_func(current_discount_percentages):
        # Constraint: product_optimized_margins >= planned_margins_per_product (with tolerance)
        # => product_optimized_margins - (planned_margins_per_product - TOLERANCE) >= 0
        _, _, _, product_optimized_margins = calculate_all_metrics_for_optimizer(current_discount_percentages)
        constraint_values = product_optimized_margins - (planned_margins_per_product - MARGIN_CONSTRAINT_TOLERANCE)
        return np.nan_to_num(constraint_values, nan=-1e9) # Penalize NaN margins heavily

    opt_options = {'disp': OPTIMIZER_VERBOSE_DEBUG, 'ftol': 1e-6, 'maxiter': 1000}
    trust_constr_opt_options = {'disp': OPTIMIZER_VERBOSE_DEBUG, 'gtol': 1e-6, 'xtol': 1e-6, 'maxiter': 1000, 'verbose': 1 if OPTIMIZER_VERBOSE_DEBUG else 0}

    constraints_slsqp = [{'type': 'ineq', 'fun': investment_constraint_func}, {'type': 'ineq', 'fun': product_margin_constraint_func}]
    method_tried = 'SLSQP'
    if OPTIMIZER_VERBOSE_DEBUG: print(f"\n--- Attempting Optimization with {method_tried} ---")
    result = minimize(objective_function, x0, method=method_tried, bounds=bounds_list, constraints=constraints_slsqp, options=opt_options)

    if OPTIMIZER_VERBOSE_DEBUG: print(f"--- {method_tried} Raw Result ---\nSuccess: {result.success}, Status: {result.status}, Message: {result.message}\nObjective: {result.fun}, Iterations: {result.nit}")

    # If SLSQP fails, try trust-constr, which can be more robust for complex constraints
    if not result.success:
        method_tried = 'trust-constr'
        if OPTIMIZER_VERBOSE_DEBUG: print(f"\n--- SLSQP failed. Attempting Optimization with {method_tried} ---")
        bounds_trust_constr = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])
        # For trust-constr, constraints need lower and upper bounds. Here, >= 0 means [0, np.inf]
        nonlinear_constraints_trust_constr = [
            NonlinearConstraint(investment_constraint_func, 0, np.inf),
            NonlinearConstraint(product_margin_constraint_func, 0, np.inf) # Each product margin must be >= its planned (with tolerance)
        ]
        try:
            result_trust = minimize(objective_function, x0, method=method_tried, bounds=bounds_trust_constr, constraints=nonlinear_constraints_trust_constr, options=trust_constr_opt_options)
            result = result_trust # Use trust-constr result if it ran
            if OPTIMIZER_VERBOSE_DEBUG: print(f"--- {method_tried} Raw Result ---\nSuccess: {result.success}, Status: {result.status}, Message: {result.message}\nObjective: {result.fun}, Iterations: {result.nit}")
        except Exception as e: # Catch potential errors during trust-constr
             if OPTIMIZER_VERBOSE_DEBUG: print(f"Error during {method_tried} optimization: {e}")
             # result remains the one from SLSQP if trust-constr errors out before producing its own result object

    if result.success:
        optimized_discounts_final = np.round(np.clip(result.x, lower_discount_pct, upper_discount_pct), 2) # Clip and round
        df_opt['Optimized Discount'] = optimized_discounts_final

        # Recalculate final metrics with the optimized (and rounded/clipped) discounts
        final_total_revenue, final_total_investment, final_total_margin, final_product_margins = calculate_all_metrics_for_optimizer(optimized_discounts_final)

        df_opt['Optimized Units'] = df_planned_static_values['Forecast Avg. Base Volume'].values * np.exp(df_planned_static_values['Discount Uplift'].values * optimized_discounts_final / 100) * df_planned_static_values['Tactic Uplift'].values
        df_opt['Optimized Per Unit Selling Price'] = np.maximum(1e-6, df_planned_static_values['List/ Base Price'].values - (df_planned_static_values['List/ Base Price'].values * optimized_discounts_final / 100))
        df_opt['Optimized Revenue'] = df_opt['Optimized Units'] * df_opt['Optimized Per Unit Selling Price']
        df_opt['Optimized Margin'] = final_product_margins # Use margins from the final calculation
        df_opt['investment'] = (df_planned_static_values['List/ Base Price'].values * optimized_discounts_final / 100) * df_opt['Optimized Units']

        # Post-optimization checks (optional, for feedback)
        if final_total_investment > max_investment * 1.01: # Allow 1% overshoot due to rounding
            if OPTIMIZER_VERBOSE_DEBUG: print(f"Warning: Final investment {format_value(final_total_investment)} slightly exceeded budget {format_value(max_investment)}.")
        # Check individual product margin constraint again after rounding
        actual_planned_margins = df_planned_static_values['Margin'].values
        violated_product_margins = df_opt['Optimized Margin'].values < (actual_planned_margins - MARGIN_CONSTRAINT_TOLERANCE - 1e-3) # Add small buffer for final check
        if np.any(violated_product_margins):
            num_violated = np.sum(violated_product_margins)
            if OPTIMIZER_VERBOSE_DEBUG: print(f"Warning: {num_violated} product(s) have optimized margin slightly below planned margin after final rounding.")
        return df_opt, None
    else:
        error_detail = f"Optimizer ({method_tried}) failed. "
        if hasattr(result, 'message') and result.message: error_detail += f"Message: '{result.message}'. "
        if hasattr(result, 'status') and result.status is not None : error_detail += f"Status code: {result.status}. "

        base_error_msg = ("Optimization failed for the selected products. This often means "
            "the constraints (budget, discount range, margin upkeep for ALL selected products) "
            "are too restrictive for the current product data (prices, costs, uplift potential)."
        )
        suggestions = ("Suggestions: \n"
            "1. Increase 'Maximum Investment' for the selected products.\n"
            "2. Widen the 'Discount Range' (especially upper bound).\n"
            "3. If goal is Revenue/Both, check if *every* selected product truly needs its margin increased/maintained (this is enforced by a constraint).\n"
            "4. Review product data: low planned margins or low uplift factors make this harder.\n"
            "5. Try optimizing fewer products or products with better uplift potential."
        )
        return pd.DataFrame(), f"{base_error_msg}\n\n{error_detail}\n\n{suggestions}"

def create_planned_vs_optimized_margin_plot(df_optimized: pd.DataFrame, df_planned_subset: pd.DataFrame) -> str:
    if df_optimized.empty or df_planned_subset.empty: return ""
    merged_df = pd.merge(
        df_planned_subset[['PL4 Sub-Category', 'Margin']].rename(columns={'Margin': 'Planned Margin'}),
        df_optimized[['PL4 Sub-Category', 'Optimized Margin']],
        on='PL4 Sub-Category', how='left'
    )
    merged_df['Optimized Margin'] = merged_df['Optimized Margin'].fillna(merged_df['Planned Margin']) # Should not happen if opt is successful for all
    plot_df = merged_df.melt(id_vars='PL4 Sub-Category', var_name='Margin Type', value_name='Margin Value')
    fig, ax = plt.subplots(figsize=(max(10, len(merged_df['PL4 Sub-Category']) * 0.5), 6))
    sns.barplot(data=plot_df, x='PL4 Sub-Category', y='Margin Value', hue='Margin Type', ax=ax, palette={'Planned Margin': 'lightcoral', 'Optimized Margin': 'mediumseagreen'})
    ax.set_title('Planned vs. Optimized Margin (Selected Products)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(fontsize=8)
    fig.tight_layout(); return save_plot_to_base64(fig)

def create_planned_vs_optimized_revenue_plot(df_optimized: pd.DataFrame, df_planned_subset: pd.DataFrame) -> str:
    if df_optimized.empty or df_planned_subset.empty: return ""
    merged_df = pd.merge(
        df_planned_subset[['PL4 Sub-Category', 'Revenue']].rename(columns={'Revenue': 'Planned Revenue'}),
        df_optimized[['PL4 Sub-Category', 'Optimized Revenue']],
        on='PL4 Sub-Category', how='left'
    )
    merged_df['Optimized Revenue'] = merged_df['Optimized Revenue'].fillna(merged_df['Planned Revenue'])
    plot_df = merged_df.melt(id_vars='PL4 Sub-Category', var_name='Revenue Type', value_name='Revenue Value')
    fig, ax = plt.subplots(figsize=(max(10, len(merged_df['PL4 Sub-Category']) * 0.5), 6))
    sns.barplot(data=plot_df, x='PL4 Sub-Category', y='Revenue Value', hue='Revenue Type', ax=ax, palette={'Planned Revenue': 'skyblue', 'Optimized Revenue': 'royalblue'})
    ax.set_title('Planned vs. Optimized Revenue (Selected Products)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(fontsize=8)
    fig.tight_layout(); return save_plot_to_base64(fig)

def create_top_bottom_performers_charts(df_optimized: pd.DataFrame, df_planned_subset: pd.DataFrame) -> Dict[str, str]:
    plots = {k: "" for k in ['revenue_increase', 'margin_increase', 'optimized_discount_applied']}
    if df_optimized.empty or df_planned_subset.empty: return plots

    df_planned_selected = df_planned_subset[['PL4 Sub-Category', 'Revenue', 'Margin']].copy()
    df_planned_selected.rename(columns={'Revenue': 'Revenue_planned', 'Margin': 'Margin_planned'}, inplace=True)

    # Ensure 'Optimized Discount' exists, even if it's all zeros (e.g., if optimization didn't change discounts)
    df_optimized_selected = df_optimized[['PL4 Sub-Category', 'Optimized Revenue', 'Optimized Margin', 'Optimized Discount']].copy()
    merged_df = pd.merge(df_planned_selected, df_optimized_selected, on='PL4 Sub-Category', how='left')

    # Handle cases where optimization might not have run for some products (should not happen with current flow)
    # or if 'Optimized Discount' wasn't set properly.
    merged_df['Optimized Revenue'].fillna(merged_df['Revenue_planned'], inplace=True)
    merged_df['Optimized Margin'].fillna(merged_df['Margin_planned'], inplace=True)
    merged_df['Optimized Discount'].fillna(0, inplace=True) # Default to 0 if missing


    merged_df['Revenue Increase %'] = np.where( np.abs(merged_df['Revenue_planned']) > 1e-6, ((merged_df['Optimized Revenue'] - merged_df['Revenue_planned']) / merged_df['Revenue_planned']) * 100, np.where(merged_df['Optimized Revenue'] > 1e-6, 1000.0, 0.0) ) # Large % if planned was 0
    merged_df['Margin Increase %'] = np.where( np.abs(merged_df['Margin_planned']) > 1e-6, ((merged_df['Optimized Margin'] - merged_df['Margin_planned']) / merged_df['Margin_planned']) * 100, np.where(merged_df['Optimized Margin'] > 1e-6, 1000.0, 0.0) )
    merged_df.fillna({'Revenue Increase %': 0, 'Margin Increase %': 0}, inplace=True) # Catch any NaNs from division by zero if new value is also zero

    num_products_to_show = min(5, len(merged_df))
    if num_products_to_show == 0: return plots


    def create_bar_subplot(data_series: pd.Series, title_suffix: str, palette_pair: Tuple[str, str], ax_pair: np.ndarray):
        if data_series.empty or num_products_to_show == 0: return

        # Ensure data_series has 'PL4 Sub-Category' as index for nlargest/nsmallest
        # This was refined in a previous step, ensure it's robust
        series_to_plot = data_series
        if series_to_plot.index.name != 'PL4 Sub-Category':
            # This assumes 'PL4 Sub-Category' is a column in merged_df from which data_series was derived
            temp_df_for_series = merged_df.set_index('PL4 Sub-Category')
            if data_series.name in temp_df_for_series.columns:
                 series_to_plot = temp_df_for_series[data_series.name]
            # If data_series was passed directly already indexed, it's fine.
            # If it's not indexed and not a column in merged_df, this plot might fail or be incorrect.
            # This part needs careful checking of how data_series is constructed before being passed.

        df_plot_top = series_to_plot.nlargest(num_products_to_show).reset_index()
        df_plot_bottom = series_to_plot.nsmallest(num_products_to_show).reset_index()

        # Check if dataframe is empty before plotting
        if not df_plot_top.empty:
            sns.barplot(data=df_plot_top, x=series_to_plot.name, y='PL4 Sub-Category', ax=ax_pair[0], palette=palette_pair[0], hue='PL4 Sub-Category', legend=False, dodge=False)
            ax_pair[0].set_title(f'Top {num_products_to_show} by {title_suffix}'); ax_pair[0].set_ylabel(''); ax_pair[0].tick_params(axis='y', labelsize=8)
        else:
            ax_pair[0].text(0.5, 0.5, "No data for Top", ha='center', va='center', transform=ax_pair[0].transAxes)


        if not df_plot_bottom.empty:
            sns.barplot(data=df_plot_bottom, x=series_to_plot.name, y='PL4 Sub-Category', ax=ax_pair[1], palette=palette_pair[1], hue='PL4 Sub-Category', legend=False, dodge=False)
            ax_pair[1].set_title(f'Bottom {num_products_to_show} by {title_suffix}'); ax_pair[1].set_ylabel(''); ax_pair[1].tick_params(axis='y', labelsize=8)
        else:
            ax_pair[1].text(0.5, 0.5, "No data for Bottom", ha='center', va='center', transform=ax_pair[1].transAxes)


    if 'Revenue Increase %' in merged_df.columns:
        fig_rev, ax_rev = plt.subplots(1, 2, figsize=(12, max(4, num_products_to_show * 0.8))); create_bar_subplot(merged_df.set_index('PL4 Sub-Category')['Revenue Increase %'], 'Revenue Increase %', ("Greens_r", "Reds_r"), ax_rev)
        fig_rev.tight_layout(); plots['revenue_increase'] = save_plot_to_base64(fig_rev)
    if 'Margin Increase %' in merged_df.columns:
        fig_mar, ax_mar = plt.subplots(1, 2, figsize=(12, max(4, num_products_to_show * 0.8))); create_bar_subplot(merged_df.set_index('PL4 Sub-Category')['Margin Increase %'], 'Margin Increase %', ("Blues_r", "Oranges_r"), ax_mar)
        fig_mar.tight_layout(); plots['margin_increase'] = save_plot_to_base64(fig_mar)
    if 'Optimized Discount' in merged_df.columns:
        fig_disc, ax_disc = plt.subplots(1, 2, figsize=(12, max(4, num_products_to_show * 0.8))); create_bar_subplot(merged_df.set_index('PL4 Sub-Category')['Optimized Discount'], 'Optimized Discount (%)', ("Purples_r", "Greys_r"), ax_disc)
        fig_disc.tight_layout(); plots['optimized_discount_applied'] = save_plot_to_base64(fig_disc)
    return plots

def create_investment_allocation_pie_chart(df_optimized: pd.DataFrame) -> str:
    if df_optimized.empty or 'investment' not in df_optimized.columns or df_optimized['investment'].sum() < 1e-6: # Check sum
        fig, ax = plt.subplots(figsize=(10, 7)); ax.text(0.5, 0.5, "No investment data available.", ha='center', va='center'); return save_plot_to_base64(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    investment_data = df_optimized.groupby('PL4 Sub-Category')['investment'].sum()
    investment_data = investment_data[investment_data > 1e-6] # Filter out negligible or zero investments
    if investment_data.empty:
        ax.text(0.5,0.5, "No significant investment to display.", ha='center', va='center'); return save_plot_to_base64(fig)

    plot_data = investment_data
    num_slices_threshold = 7 # Max slices before grouping into "Others"
    if len(investment_data) > num_slices_threshold:
        top_n = investment_data.nlargest(num_slices_threshold -1) # Keep N-1 largest
        others_sum = investment_data[~investment_data.index.isin(top_n.index)].sum()
        if others_sum > 1e-6: # Only add "Others" if it's significant
            plot_data = pd.concat([top_n, pd.Series([others_sum], index=['Others (Combined)'])])
        else:
            plot_data = top_n # If "Others" is negligible, just show top N-1

    explode_values = None
    if len(plot_data) > 1: # Explode only if there's more than one slice
        largest_slice_idx = plot_data.values.argmax()
        explode_values = [0.05 if i == largest_slice_idx else 0 for i in range(len(plot_data))]

    wedges, texts, autotexts = ax.pie(plot_data, labels=None, autopct='%1.1f%%', startangle=140,
                                      wedgeprops=dict(width=0.45), pctdistance=0.80,
                                      explode=explode_values, textprops={'fontsize': 9, 'fontweight':'bold', 'color':'white'})
    for autotext in autotexts:
        autotext.set_horizontalalignment('center'); autotext.set_verticalalignment('center') # Center percentage text

    ax.set_title('Investment Allocation (Selected Products)', fontsize=14, pad=15)
    ax.legend(wedges, plot_data.index, title="Products", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize='small', title_fontsize='medium')
    fig.tight_layout(rect=[0, 0, 0.82, 0.95]); # Adjust layout to make space for legend
    return save_plot_to_base64(fig)

def create_cumulative_growth_curve(df_planned_subset: pd.DataFrame) -> str:
    if df_planned_subset.empty: return ""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    discount_levels = np.linspace(0, 50, 11) # 0% to 50% discount in 11 steps
    total_rev_points, total_marg_points = [], []
    currency_s = format_value(0, kind='currency')[0] # Get currency symbol

    # Vectorized calculations
    base_vol = df_planned_subset['Forecast Avg. Base Volume'].astype(float).values
    uplift = df_planned_subset['Discount Uplift'].astype(float).values
    tactic_up = df_planned_subset['Tactic Uplift'].astype(float).values
    list_price = df_planned_subset['List/ Base Price'].astype(float).values
    cogs = df_planned_subset['Per Unit COGS'].astype(float).values

    for d_pct in discount_levels:
        d_ratio = d_pct / 100.0
        current_units = base_vol * np.exp(uplift * d_ratio) * tactic_up
        current_abs_disc_value = list_price * d_ratio
        current_psp = list_price - current_abs_disc_value
        current_rev_total = np.sum(current_units * current_psp)
        current_marg_total = current_rev_total - np.sum(current_units * cogs)
        total_rev_points.append(current_rev_total)
        total_marg_points.append(current_marg_total)

    ax1.set_xlabel('Uniform Discount (%)', fontsize=10)
    ax1.set_ylabel(f'Total Revenue ({currency_s})', color='tab:blue', fontsize=10)
    ax1.plot(discount_levels, total_rev_points, color='tab:blue', marker='o', linestyle='-', lw=1.5, ms=4, label='Total Revenue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=8); ax1.tick_params(axis='x', labelsize=8)

    ax2 = ax1.twinx() # Share the same x-axis
    ax2.set_ylabel(f'Total Margin ({currency_s})', color='tab:green', fontsize=10)
    ax2.plot(discount_levels, total_marg_points, color='tab:green', marker='x', linestyle='--', lw=1.5, ms=4, label='Total Margin')
    ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=8)

    fig.suptitle('Hypothetical Growth for Selected Products (Uniform Discount)', fontsize=12)
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8) # Legend below plot
    fig.tight_layout(rect=[0, 0.1, 1, 0.93]); # Adjust layout for suptitle and legend
    return save_plot_to_base64(fig)

def create_roi_by_product_chart(df_optimized: pd.DataFrame, df_planned_subset: pd.DataFrame) -> str:
    if df_optimized.empty or df_planned_subset.empty: return ""

    df_planned_selected_roi = df_planned_subset[['PL4 Sub-Category', 'Margin']].copy().rename(columns={'Margin': 'Margin_planned'})

    # Ensure necessary columns exist in df_optimized, defaulting if not (e.g., if optimization failed or didn't run)
    if 'investment' not in df_optimized.columns: df_optimized['investment'] = 0
    if 'Optimized Margin' not in df_optimized.columns:
        # If Optimized Margin is missing, try to use planned margin for ROI calculation (ROI would be 0 or infinite)
        # This case should ideally be handled by ensuring df_optimized always has this column post-optimization attempt.
        df_optimized = pd.merge(df_optimized, df_planned_selected_roi, on='PL4 Sub-Category', how='left')
        df_optimized.rename(columns={'Margin_planned': 'Optimized Margin'}, inplace=True) # Use planned as proxy

    df_optimized_selected_roi = df_optimized[['PL4 Sub-Category', 'Optimized Margin', 'investment']].copy()
    roi_df = pd.merge(df_planned_selected_roi, df_optimized_selected_roi, on='PL4 Sub-Category', how='left')

    # Fill NaNs that might occur if a product in planned_subset wasn't in optimized_subset
    roi_df['Optimized Margin'].fillna(roi_df['Margin_planned'], inplace=True)
    roi_df['investment'].fillna(0, inplace=True)

    roi_df['Margin Gain'] = roi_df['Optimized Margin'] - roi_df['Margin_planned']
    # Calculate ROI: (Margin Gain / Investment) * 100
    # Handle division by zero: if investment is zero, ROI is infinite if gain > 0, else 0.
    roi_df['ROI'] = np.where(
        np.abs(roi_df['investment']) > 1e-6, # If investment is not zero
        (roi_df['Margin Gain'] / roi_df['investment']) * 100,
        np.where(roi_df['Margin Gain'] > 1e-6, 2000.0, 0.0) # Arbitrarily large ROI if gain > 0 and inv = 0, else 0
    )
    roi_df.fillna({'ROI': 0}, inplace=True) # Catch any other NaNs

    roi_df_sorted = roi_df.sort_values(by='ROI', ascending=False)
    if roi_df_sorted.empty:
        fig, ax = plt.subplots(figsize=(8,5)); ax.text(0.5,0.5,"No ROI data available.", ha='center'); return save_plot_to_base64(fig)

    num_products_to_show = min(15, len(roi_df_sorted)) # Show top 15 or fewer
    roi_df_to_plot = roi_df_sorted.head(num_products_to_show)

    if roi_df_to_plot.empty or roi_df_to_plot['ROI'].isnull().all(): # check if any data to plot
        fig, ax = plt.subplots(figsize=(8,5)); ax.text(0.5,0.5,"No significant ROI data to plot.", ha='center'); return save_plot_to_base64(fig)


    fig, ax = plt.subplots(figsize=(10, max(5, len(roi_df_to_plot) * 0.35))) # Dynamic height
    sns.barplot(data=roi_df_to_plot, x='ROI', y='PL4 Sub-Category', hue='PL4 Sub-Category', palette="viridis", ax=ax, legend=False, dodge=False)
    ax.set_title(f'ROI by Product (Top {num_products_to_show} Selected)', fontsize=14)
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    fig.tight_layout(); return save_plot_to_base64(fig)

def create_volume_margin_tradeoff_bubble_chart(df_optimized: pd.DataFrame, df_planned_subset: pd.DataFrame) -> str:
    if df_optimized.empty or df_planned_subset.empty: return ""

    df_p = df_planned_subset[['PL4 Sub-Category', 'Units', 'Margin', 'Revenue']].rename(columns={'Units':'U_p','Margin':'M_p','Revenue':'R_p'})
    # Ensure df_optimized has the necessary columns
    opt_cols_needed = ['PL4 Sub-Category', 'Optimized Units', 'Optimized Margin', 'Optimized Revenue']
    for col in opt_cols_needed:
        if col not in df_optimized.columns: # If a column is missing, create it by copying from planned (as a fallback)
            if col == 'Optimized Units': df_optimized[col] = df_p['U_p']
            elif col == 'Optimized Margin': df_optimized[col] = df_p['M_p']
            elif col == 'Optimized Revenue': df_optimized[col] = df_p['R_p']
            elif col == 'PL4 Sub-Category' and 'PL4 Sub-Category' not in df_optimized.columns: # Should not happen
                 df_optimized['PL4 Sub-Category'] = df_p['PL4 Sub-Category']


    df_o = df_optimized[opt_cols_needed].rename(columns={'Optimized Units':'U_o','Optimized Margin':'M_o','Optimized Revenue':'R_o'})
    tdf = pd.merge(df_p, df_o, on='PL4 Sub-Category', how='left')

    # Fill NaNs that might occur if a product wasn't in optimized results (should use planned values)
    tdf['U_o'].fillna(tdf['U_p'], inplace=True)
    tdf['M_o'].fillna(tdf['M_p'], inplace=True)
    tdf['R_o'].fillna(tdf['R_p'], inplace=True)


    tdf['Volume Increase %'] = np.where(np.abs(tdf['U_p']) > 1e-6, ((tdf['U_o'] - tdf['U_p']) / tdf['U_p']) * 100, np.where(tdf['U_o'] > 1e-6, 1000.0, 0.0))
    tdf['Margin Increase %'] = np.where(np.abs(tdf['M_p']) > 1e-6, ((tdf['M_o'] - tdf['M_p']) / tdf['M_p']) * 100, np.where(tdf['M_o'] > 1e-6, 1000.0, 0.0))
    tdf.fillna({'Volume Increase %': 0, 'Margin Increase %': 0, 'R_o': 0}, inplace=True)

    if tdf.empty:
        fig, ax = plt.subplots(figsize=(10,7)); ax.text(0.5,0.5,"No data for trade-off plot.", ha='center'); return save_plot_to_base64(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    sizes = tdf['R_o'].copy()
    sizes[sizes < 0] = 0 # Bubble sizes should be non-negative
    min_val_size, max_val_size = sizes.min(), sizes.max()
    min_bubble_sz, max_bubble_sz = 30, 800 # Min/max pixel size for bubbles

    if pd.isna(min_val_size) or pd.isna(max_val_size): # Handle all NaN case for sizes
        sizes_norm = pd.Series([min_bubble_sz]*len(tdf), index=tdf.index)
    elif max_val_size > min_val_size : # Avoid division by zero if all revenue values are the same
        sizes_norm = min_bubble_sz + (sizes - min_val_size) / (max_val_size - min_val_size) * (max_bubble_sz - min_bubble_sz)
    else: # All bubbles get min size if all revenues are same (or only one product)
         sizes_norm = pd.Series([min_bubble_sz]*len(tdf), index=tdf.index)

    sizes_norm.fillna(min_bubble_sz, inplace=True) # Handle any NaNs in sizes_norm itself

    # Ensure PL4 Sub-Category is present for hue
    if 'PL4 Sub-Category' not in tdf.columns:
        tdf['PL4 Sub-Category'] = "Unknown Product"


    sns.scatterplot(data=tdf, x='Volume Increase %', y='Margin Increase %',
                    size=sizes_norm, hue='PL4 Sub-Category', alpha=0.7, ax=ax,
                    legend="auto", sizes=(min_bubble_sz, max_bubble_sz)) # Pass tuple to sizes

    ax.set_title('Volume vs. Margin Trade-off (Selected Products)', fontsize=14)
    ax.axhline(0,c='grey',ls='--'); ax.axvline(0,c='grey',ls='--')
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    # Filter out legend entries for 'size' (which are typically numeric)
    # and specific keywords that might relate to size rather than hue.
    hue_handles = [h for i, h in enumerate(handles) if not labels[i].replace('.', '', 1).isdigit() and labels[i] not in ['R_o', 'Optimized Revenue', 'size', str(min_bubble_sz), str(max_bubble_sz)]]
    hue_labels =  [l for i, l in enumerate(labels)  if not labels[i].replace('.', '', 1).isdigit() and labels[i] not in ['R_o', 'Optimized Revenue', 'size', str(min_bubble_sz), str(max_bubble_sz)]]


    if hue_handles:
        ax.legend(hue_handles, hue_labels, title='Product', bbox_to_anchor=(1.02,1), loc='upper left', fontsize='small', title_fontsize='medium')
    elif ax.get_legend() is not None: # If a legend exists but we filtered all items
        ax.get_legend().set_visible(False)


    fig.tight_layout(rect=[0,0,0.83,0.95]); # Adjust for legend
    return save_plot_to_base64(fig)


# --- Data Generation (Called after function definitions) ---
tea_product_names = ['Assam Gold Leaf Tea', 'Darjeeling First Flush', 'Nilgiri Frost Tea', 'Masala Chai Premium Blend','Herbal Infusion Relax', 'Green Tea Classic', 'Earl Grey Imperial', 'Oolong Select Reserve','White Peony Delicate', 'Rooibos Vanilla Spice', 'Pu-erh Aged Brick', 'Matcha Ceremonial Grade','Chamomile Soothe', 'Ginger Turmeric Zest']
real_tea_prices = [299.0, 450.0, 320.0, 250.0, 199.0, 180.0, 350.0, 420.0, 550.0, 220.0, 600.0, 750.0, 150.0, 190.0]
data1 = generate_category_data("Tea", 14, (500, 2500), (1.8, 3.5), (1.5, 4.0), tea_product_names, real_tea_prices)

poha_product_names = ['Classic Poha Original', 'Thick Poha Gold', 'Thin Poha Select', 'Classic Poha Supreme', 'Red Rice Poha Classic', 'Organic Poha Flakes', 'Brown Rice Poha', 'Fine Poha Standard', 'Quick-Cook Poha', 'Jumbo Poha Flakes', 'Gluten-Free Poha', 'Iron-Fortified Poha', 'Multi-Grain Poha', 'Sprouted Poha Mix']
real_poha_prices = [45.0, 60.0, 50.0, 70.0, 80.0, 90.0, 85.0, 40.0, 55.0, 75.0, 95.0, 65.0, 110.0, 120.0]
data2 = generate_category_data("Poha", 14, (2000, 10000), (1.4, 2.2), (2.0, 5.0), poha_product_names, real_poha_prices)

coffee_product_names = ['Espresso Intenso Beans', 'Arabica Gold Filter Coffee', 'Robusta Dark Roast Powder', 'Instant Classic Blend', 'Decaf Smooth Roast', 'Single Origin Colombian', 'Flavored Hazelnut Coffee', 'Cold Brew Concentrate', 'Organic Fair-Trade Beans', 'Chicory Blend South Indian', 'Luxury Blue Mountain', 'Vietnamese Style Phin Filter', 'Coffee Pods Variety Pack', 'Ready-to-Drink Iced Latte']
real_coffee_prices = [350.0, 280.0, 220.0, 150.0, 300.0, 400.0, 250.0, 180.0, 450.0, 120.0, 800.0, 320.0, 500.0, 90.0]
data3 = generate_category_data("Coffee", 14, (800, 3000), (1.7, 4.0), (1.2, 3.5), coffee_product_names, real_coffee_prices)

spices_product_names = ['Turmeric Powder Premium', 'Red Chilli Powder Extra Hot', 'Coriander Powder Aromatic', 'Cumin Seeds Whole', 'Garam Masala Royal Blend', 'Black Pepper Ground', 'Cardamom Green Whole', 'Cinnamon Sticks Ceylon', 'Asafoetida (Hing) Strong', 'Sambar Powder Special', 'Mustard Seeds Black', 'Fenugreek Seeds', 'Bay Leaves Whole', 'Cloves Whole Best Quality']
real_spices_prices = [50.0, 60.0, 45.0, 70.0, 120.0, 80.0, 250.0, 150.0, 90.0, 75.0, 40.0, 35.0, 30.0, 180.0]
data4 = generate_category_data("Spices", 14, (1000, 7000), (1.9, 5.0), (1.8, 4.5), spices_product_names, real_spices_prices)

masala_mix_names = ['Peri Peri Fries Sprinkle', 'Schezwan Fried Rice Mix', 'Instant Paneer Butter Masala', 'Chili Chicken Quick Mix', 'Pasta Masala Magic', 'Chicken 65 Marinade', 'Hakka Noodle Seasoning', 'Mutter Paneer Gravy Base', 'Shahi Paneer Ready Mix', 'Chinese Manchurian Mix', 'Chowmein Masala Burst', 'Veg Biryani Masala Kit', 'Tandoori Tikka Masala', 'Fish Curry Goan Style Mix']
real_masala_mix_prices = [30.0, 40.0, 55.0, 50.0, 25.0, 60.0, 35.0, 45.0, 65.0, 40.0, 30.0, 70.0, 60.0, 75.0]
data5 = generate_category_data("Masala", 14, (1500, 8000), (1.6, 2.8), (2.2, 5.5), masala_mix_names, real_masala_mix_prices) # Corrected Category name

Tea_df = pd.DataFrame(data1)
Poha_df = pd.DataFrame(data2)
Coffee_df = pd.DataFrame(data3)
Spices_df = pd.DataFrame(data4)
Masala_df = pd.DataFrame(data5)

category_dict = {
    "Tea": Tea_df, "Poha": Poha_df, "Coffee": Coffee_df,
    "Spices": Spices_df, "Masala": Masala_df
}

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    session_id = str(uuid.uuid4())
    request.session["session_id"] = session_id
    request.session.pop('session_error_message', None)

    sessions[session_id] = {
        'df_full_category_plan': None,
        'df_planned_subset': None,
        'df_optimized_subset': None,
        'tf_planned_totals_subset': create_total_df_structure(),
        'tf_final_comparison_subset': None,
        'tf_full_category_comparison': None,
        'params': {},
        'category_name': None,
        'selected_product_names': [],
    }
    error_message_from_query = request.query_params.get("error_message")
    if error_message_from_query:
        request.session['session_error_message'] = error_message_from_query

    return templates.TemplateResponse("index.html", {
        "request": request,
        "session_id": session_id,
        "session_error_message": request.session.get('session_error_message')
    })

def _prepare_product_selection_data(request: Request, session_id: str, category: str):
    if session_id not in sessions:
        request.session['session_error_message'] = "Invalid session. Please start over."
        return None, RedirectResponse(f"/?error_message=Invalid session", status_code=303)

    current_app_session = sessions[session_id]
    df_category_full = category_dict.get(category)

    if df_category_full is None:
        request.session['session_error_message'] = f"Category '{category}' not found."
        return None, RedirectResponse(f"/?session_id={session_id}&error_message=Category not found", status_code=303)

    current_app_session['category_name'] = category
    current_app_session['df_full_category_plan'] = df_category_full.copy()
    current_app_session['df_planned_subset'] = None
    current_app_session['df_optimized_subset'] = None
    current_app_session['selected_product_names'] = []

    current_params = current_app_session.get('params', {}).copy()
    if not current_params or current_params.get('category') != category:
        current_params = {
            'max_investment': 10000, 'lower_discount': 0,
            'upper_discount': 10, 'optimization_goal':'Revenue'
        }
    current_params['category'] = category
    current_app_session['params'] = current_params

    product_list = df_category_full['PL4 Sub-Category'].tolist()
    request.session.pop('session_error_message', None) # Clear before rendering this page

    template_data = {
        "request": request, "session_id": session_id,
        "category_name": category, "product_list": product_list,
        "current_params": current_params,
        "session_error_message": None
    }
    return template_data, None

@app.post("/show_product_selection", response_class=HTMLResponse)
async def show_product_selection_post(request: Request, session_id: str = Form(...), category: str = Form(...)):
    context_data, redirect_response = _prepare_product_selection_data(request, session_id, category)
    if redirect_response:
        return redirect_response
    return templates.TemplateResponse("product_selection.html", context_data)

@app.get("/go_to_product_selection", response_class=HTMLResponse)
async def show_product_selection_get(request: Request, session_id: str = Query(...), category: str = Query(...)):
    context_data, redirect_response = _prepare_product_selection_data(request, session_id, category)
    if redirect_response:
        return redirect_response
    return templates.TemplateResponse("product_selection.html", context_data)


# ... (other parts of your main.py file) ...

@app.post("/prepare_plan_view", response_class=HTMLResponse)
async def prepare_plan_view(request: Request, session_id: str = Form(...),
                            selected_products: List[str] = Form([]),
                            max_investment: float = Form(...),
                            lower_discount: float = Form(...),
                            upper_discount: float = Form(...),
                            optimization_goal: str = Form(...)):
    if session_id not in sessions:
        request.session['session_error_message'] = "Invalid session. Please start over."
        return RedirectResponse(f"/?error_message=Invalid session", status_code=303)

    current_app_session = sessions[session_id]
    category_from_session = current_app_session.get('category_name')

    if not selected_products:
        error_message = "No products selected for optimization. Please select at least one product."
        request.session['session_error_message'] = error_message
        if category_from_session:
            context_data, _ = _prepare_product_selection_data(request, session_id, category_from_session)
            if context_data:
                context_data['current_params'].update({
                    'max_investment': max_investment, 'lower_discount': lower_discount,
                    'upper_discount': upper_discount, 'optimization_goal': optimization_goal
                })
                context_data['session_error_message'] = error_message
                return templates.TemplateResponse("product_selection.html", context_data, status_code=400)
        return RedirectResponse(f"/?session_id={session_id}&error_message={error_message}", status_code=303)

    df_full_category_plan = current_app_session.get('df_full_category_plan')
    if df_full_category_plan is None or category_from_session is None:
        request.session['session_error_message'] = "Session data missing (category or plan). Please start over."
        return RedirectResponse(f"/?session_id={session_id}&error_message=Session data missing", status_code=303)

    df_planned_subset = df_full_category_plan[df_full_category_plan['PL4 Sub-Category'].isin(selected_products)].copy()
    if df_planned_subset.empty:
        error_message = "Selected products not found in category data. Please re-select."
        request.session['session_error_message'] = error_message
        if category_from_session:
            context_data, _ = _prepare_product_selection_data(request, session_id, category_from_session)
            if context_data:
                context_data['current_params'].update({
                    'max_investment': max_investment, 'lower_discount': lower_discount,
                    'upper_discount': upper_discount, 'optimization_goal': optimization_goal
                })
                context_data['session_error_message'] = error_message
                return templates.TemplateResponse("product_selection.html", context_data, status_code=400)
        return RedirectResponse(f"/?session_id={session_id}&error_message={error_message}", status_code=303)

    current_app_session['df_planned_subset'] = df_planned_subset
    current_app_session['selected_product_names'] = selected_products
    user_params = {'max_investment': max_investment, 'lower_discount': lower_discount,
                   'upper_discount': upper_discount, 'category': category_from_session,
                   'optimization_goal': optimization_goal,
                   'num_selected_products': len(selected_products)}
    current_app_session['params'] = user_params

    tf_planned_subset_totals = create_total_df_structure()
    tf_planned_subset_totals.loc[tf_planned_subset_totals['Data-Totals'] == 'Units', 'Planned'] = df_planned_subset['Units'].sum()
    tf_planned_subset_totals.loc[tf_planned_subset_totals['Data-Totals'] == 'Revenue', 'Planned'] = df_planned_subset['Revenue'].sum()
    tf_planned_subset_totals.loc[tf_planned_subset_totals['Data-Totals'] == 'Margins', 'Planned'] = df_planned_subset['Margin'].sum()
    tf_planned_subset_totals.loc[tf_planned_subset_totals['Data-Totals'] == 'Investment', 'Planned'] = df_planned_subset['Dollor Investment'].sum()
    current_app_session['tf_planned_totals_subset'] = tf_planned_subset_totals

    # --- MODIFICATION HERE ---
    # selected_columns_display = ['PL4 Sub-Category', 'Forecast Avg. Base Volume', 'Units', 'Revenue', 'Margin', 'Discount']
    selected_columns_display = ['PL4 Sub-Category', 'Units', 'Revenue', 'Margin', 'Discount'] # Removed 'Forecast Avg. Base Volume'
    # --- END OF MODIFICATION ---

    data_to_display = []
    for _, row in df_planned_subset[selected_columns_display].iterrows(): # df_planned_subset might still contain it, but we only select display columns
        record = {col: row.get(col) for col in selected_columns_display}
        record['Units'] = format_value(record.get('Units'), kind="units")
        record['Revenue'] = format_value(record.get('Revenue'), kind="currency")
        record['Margin'] = format_value(record.get('Margin'), kind="currency")
        record['Discount'] = format_value(record.get('Discount', 0.0), kind="percentage")
        # 'Forecast Avg. Base Volume' formatting removed as it's not in selected_columns_display
        # record['Forecast Avg. Base Volume'] = format_value(record.get('Forecast Avg. Base Volume'), kind="units") 
        data_to_display.append(record)

    request.session.pop('session_error_message', None)

    return templates.TemplateResponse("optimization.html", {
        "request": request, "data": data_to_display,
        "columns": selected_columns_display, # This list now excludes the column
        "category_display_name": f"{category_from_session} (Selected Products)",
        "session_id": session_id, "params": user_params,
        "session_error_message": None
    })



@app.post("/optimize_run", response_class=HTMLResponse)
async def optimize_run(request: Request, session_id: str = Form(...),
                    max_investment: float = Form(...),
                    lower_discount: float = Form(...),
                    upper_discount: float = Form(...),
                    optimization_goal: str = Form(...)):
    if session_id not in sessions:
        request.session['session_error_message'] = "Invalid session. Please start over."
        return RedirectResponse(f"/?error_message=Invalid session", status_code=303)

    current_app_session = sessions[session_id]
    if current_app_session.get('df_planned_subset') is None:
        request.session['session_error_message'] = "Planned data not found. Please select products again."
        category_name = current_app_session.get('category_name')
        if category_name:
            return RedirectResponse(f"/go_to_product_selection?session_id={session_id}&category={category_name}&error_message=Planned data missing", status_code=303)
        return RedirectResponse(f"/?session_id={session_id}&error_message=Planned data missing", status_code=303)

    df_planned_subset_from_session = current_app_session['df_planned_subset'].copy()
    category_name = current_app_session['category_name']
    current_params_for_run = current_app_session.get('params', {}).copy()
    current_params_for_run.update({
        'max_investment': max_investment, 'lower_discount': lower_discount,
        'upper_discount': upper_discount, 'optimization_goal': optimization_goal,
        'category': category_name
    })
    current_app_session['params'] = current_params_for_run

    df_optimized_subset, error_msg = optimize_discounts(
        df_input_planned=df_planned_subset_from_session,
        max_investment=max_investment,
        lower_discount_pct=lower_discount,
        upper_discount_pct=upper_discount,
        optimization_goal=optimization_goal
    )

    if error_msg or df_optimized_subset.empty:
        selected_columns_display = ['PL4 Sub-Category', 'Forecast Avg. Base Volume', 'Units', 'Revenue', 'Margin', 'Discount']
        data_to_display_on_error = []
        for _, row in df_planned_subset_from_session[selected_columns_display].iterrows():
            record = {col: row.get(col) for col in selected_columns_display}
            record['Units'] = format_value(record.get('Units'), kind="units")
            record['Revenue'] = format_value(record.get('Revenue'), kind="currency")
            record['Margin'] = format_value(record.get('Margin'), kind="currency")
            record['Discount'] = format_value(record.get('Discount', 0.0), kind="percentage")
            record['Forecast Avg. Base Volume'] = format_value(record.get('Forecast Avg. Base Volume'), kind="units")
            data_to_display_on_error.append(record)

        return templates.TemplateResponse("optimization.html", {
            "request": request, "data": data_to_display_on_error,
            "columns": selected_columns_display,
            "category_display_name": f"{category_name} (Selected Products)",
            "error": error_msg or "Optimization failed to find a solution.",
            "session_id": session_id, "params": current_params_for_run,
            "session_error_message": None
        })

    current_app_session['df_optimized_subset'] = df_optimized_subset
    tf_comparison_subset = current_app_session['tf_planned_totals_subset'].copy()
    tf_comparison_subset.loc[tf_comparison_subset['Data-Totals'] == 'Units', 'Optimized'] = df_optimized_subset['Optimized Units'].sum()
    tf_comparison_subset.loc[tf_comparison_subset['Data-Totals'] == 'Revenue', 'Optimized'] = df_optimized_subset['Optimized Revenue'].sum()
    tf_comparison_subset.loc[tf_comparison_subset['Data-Totals'] == 'Margins', 'Optimized'] = df_optimized_subset['Optimized Margin'].sum()
    tf_comparison_subset.loc[tf_comparison_subset['Data-Totals'] == 'Investment', 'Optimized'] = df_optimized_subset['investment'].sum()

    planned_v_sub = tf_comparison_subset['Planned'].values.astype(float)
    optimized_v_sub = tf_comparison_subset['Optimized'].values.astype(float)
    percent_changes_sub = []
    for i in range(len(planned_v_sub)):
        if abs(planned_v_sub[i]) < 1e-9: percent_changes_sub.append('inf' if abs(optimized_v_sub[i]) > 1e-9 else 0.0)
        else: percent_changes_sub.append(((optimized_v_sub[i] - planned_v_sub[i]) / planned_v_sub[i]) * 100)
    tf_comparison_subset['% Change'] = percent_changes_sub
    current_app_session['tf_final_comparison_subset'] = tf_comparison_subset

    tf_subset_display_list = []
    for _, rec_row in tf_comparison_subset.iterrows():
        rec = rec_row.to_dict(); fmt_rec = rec.copy()
        kind = "units" if rec['Data-Totals'] == 'Units' else "currency"
        fmt_rec['Planned'] = format_value(rec['Planned'], kind=kind); fmt_rec['Optimized'] = format_value(rec['Optimized'], kind=kind)
        if rec['% Change'] == 'inf': fmt_rec['% Change'] = "∞ %"
        elif pd.notna(rec['% Change']) and rec['% Change'] != '': fmt_rec['% Change'] = format_value(float(rec['% Change']), kind="percentage")
        else: fmt_rec['% Change'] = ''
        tf_subset_display_list.append(fmt_rec)

    df_full_category_plan_orig = current_app_session['df_full_category_plan'].copy() # Original full plan
    df_full_category_optimized_state = df_full_category_plan_orig.copy() # Start with original plan

    # Index for efficient update
    if 'PL4 Sub-Category' in df_full_category_optimized_state.columns:
        df_full_category_optimized_state.set_index('PL4 Sub-Category', inplace=True, drop=False) # Keep column
    
    df_optimized_subset_for_update = df_optimized_subset.copy()
    if 'PL4 Sub-Category' in df_optimized_subset_for_update.columns:
         df_optimized_subset_for_update.set_index('PL4 Sub-Category', inplace=True, drop=False)


    # Columns to update from optimized_subset to full_category_optimized_state
    update_cols_map = {
        'Optimized Units': 'Units',
        'Optimized Revenue': 'Revenue',
        'Optimized Margin': 'Margin',
        'Optimized Discount': 'Discount',
        'investment': 'Dollor Investment',
        'Optimized Per Unit Selling Price': 'Per Unit Selling Price'
    }

    for opt_col_name, full_plan_col_name in update_cols_map.items():
        if opt_col_name in df_optimized_subset_for_update.columns:
            # Create a series from the optimized subset for the current column to update
            update_series = df_optimized_subset_for_update[opt_col_name]
            
            # Ensure the target column in df_full_category_optimized_state can accept the data type
            if full_plan_col_name in df_full_category_optimized_state.columns:
                target_dtype = df_full_category_optimized_state[full_plan_col_name].dtype
                if pd.api.types.is_numeric_dtype(update_series.dtype) and \
                   not pd.api.types.is_dtype_equal(update_series.dtype, target_dtype):
                    if pd.api.types.is_float_dtype(update_series.dtype) and \
                       pd.api.types.is_integer_dtype(target_dtype):
                        df_full_category_optimized_state[full_plan_col_name] = df_full_category_optimized_state[full_plan_col_name].astype(float)
            
            # Perform the update using .loc for safety with mixed types or potential new columns
            for idx, val in update_series.items():
                if idx in df_full_category_optimized_state.index:
                     df_full_category_optimized_state.loc[idx, full_plan_col_name] = val
                # else: # Product in optimized subset not in full category plan - should not happen
                #    print(f"Warning: Product {idx} from optimized results not found in full category plan for update.")


    # Reset index if it was set
    if isinstance(df_full_category_optimized_state.index, pd.Index) and df_full_category_optimized_state.index.name == 'PL4 Sub-Category':
        df_full_category_optimized_state.reset_index(drop=True, inplace=True) # drop=True if PL4... is already a column

    tf_full_cat_comp = create_full_category_total_df_structure()
    tf_full_cat_comp.loc[tf_full_cat_comp['Data-Totals'] == 'Units', 'Planned'] = df_full_category_plan_orig['Units'].sum()
    tf_full_cat_comp.loc[tf_full_cat_comp['Data-Totals'] == 'Revenue', 'Planned'] = df_full_category_plan_orig['Revenue'].sum()
    tf_full_cat_comp.loc[tf_full_cat_comp['Data-Totals'] == 'Margins', 'Planned'] = df_full_category_plan_orig['Margin'].sum()
    tf_full_cat_comp.loc[tf_full_cat_comp['Data-Totals'] == 'Investment', 'Planned'] = df_full_category_plan_orig['Dollor Investment'].sum()

    tf_full_cat_comp.loc[tf_full_cat_comp['Data-Totals'] == 'Units', 'Optimized State'] = df_full_category_optimized_state['Units'].sum()
    tf_full_cat_comp.loc[tf_full_cat_comp['Data-Totals'] == 'Revenue', 'Optimized State'] = df_full_category_optimized_state['Revenue'].sum()
    tf_full_cat_comp.loc[tf_full_cat_comp['Data-Totals'] == 'Margins', 'Optimized State'] = df_full_category_optimized_state['Margin'].sum()
    tf_full_cat_comp.loc[tf_full_cat_comp['Data-Totals'] == 'Investment', 'Optimized State'] = df_full_category_optimized_state['Dollor Investment'].sum()


    planned_v_full = tf_full_cat_comp['Planned'].values.astype(float)
    optimized_v_full = tf_full_cat_comp['Optimized State'].values.astype(float)
    percent_changes_full = []
    for i in range(len(planned_v_full)):
        if abs(planned_v_full[i]) < 1e-9: percent_changes_full.append('inf' if abs(optimized_v_full[i]) > 1e-9 else 0.0)
        else: percent_changes_full.append(((optimized_v_full[i] - planned_v_full[i]) / planned_v_full[i]) * 100)
    tf_full_cat_comp['% Change'] = percent_changes_full
    current_app_session['tf_full_category_comparison'] = tf_full_cat_comp

    tf_full_cat_display_list = []
    for _, rec_row_fc in tf_full_cat_comp.iterrows():
        rec_fc = rec_row_fc.to_dict(); fmt_rec_fc = rec_fc.copy()
        kind_fc = "units" if rec_fc['Data-Totals'] == 'Units' else "currency"
        fmt_rec_fc['Planned'] = format_value(rec_fc['Planned'], kind=kind_fc); fmt_rec_fc['Optimized State'] = format_value(rec_fc['Optimized State'], kind=kind_fc)
        if rec_fc['% Change'] == 'inf': fmt_rec_fc['% Change'] = "∞ %"
        elif pd.notna(rec_fc['% Change']) and rec_fc['% Change'] != '': fmt_rec_fc['% Change'] = format_value(float(rec_fc['% Change']), kind="percentage")
        else: fmt_rec_fc['% Change'] = ''
        tf_full_cat_display_list.append(fmt_rec_fc)

    df_planned_subset_for_results = df_planned_subset_from_session[['PL4 Sub-Category', 'Units', 'Revenue', 'Margin', 'Discount']].rename(
        columns={'Units': 'Planned Units', 'Revenue': 'Planned Revenue', 'Margin': 'Planned Margin', 'Discount': 'Planned Discount (%)'}
    )
    df_results_display_subset = pd.merge(df_optimized_subset, df_planned_subset_for_results, on='PL4 Sub-Category', how='left')
    optimized_display_columns = ['PL4 Sub-Category', 'Planned Units', 'Optimized Units', 'Planned Revenue', 'Optimized Revenue', 'Planned Margin', 'Optimized Margin', 'Planned Discount (%)', 'Optimized Discount', 'investment']
    for col in optimized_display_columns:
        if col not in df_results_display_subset.columns: df_results_display_subset[col] = 0 if 'Units' in col or 'Revenue' in col or 'Margin' in col or 'investment' in col else (0.0 if 'Discount' in col else "N/A")

    formatted_optimized_data_list = []
    for _, rec_row_opt in df_results_display_subset[optimized_display_columns].iterrows():
        rec_opt = rec_row_opt.to_dict(); new_rec = {k: rec_opt.get(k) for k in optimized_display_columns}
        new_rec['Planned Units'] = format_value(new_rec.get('Planned Units'), kind="units"); new_rec['Optimized Units'] = format_value(new_rec.get('Optimized Units'), kind="units")
        new_rec['Planned Revenue'] = format_value(new_rec.get('Planned Revenue'), kind="currency"); new_rec['Optimized Revenue'] = format_value(new_rec.get('Optimized Revenue'), kind="currency")
        new_rec['Planned Margin'] = format_value(new_rec.get('Planned Margin'), kind="currency"); new_rec['Optimized Margin'] = format_value(new_rec.get('Optimized Margin'), kind="currency")
        new_rec['Planned Discount (%)'] = format_value(new_rec.get('Planned Discount (%)', 0.0), kind="percentage"); new_rec['Optimized Discount'] = format_value(new_rec.get('Optimized Discount', 0.0), kind="percentage")
        new_rec['investment'] = format_value(new_rec.get('investment'), kind="currency")
        formatted_optimized_data_list.append(new_rec)

    plot_data = {
        'plot_planned_vs_optimized_margin': create_planned_vs_optimized_margin_plot(df_optimized_subset, df_planned_subset_from_session),
        'plot_planned_vs_optimized_revenue': create_planned_vs_optimized_revenue_plot(df_optimized_subset, df_planned_subset_from_session),
        **create_top_bottom_performers_charts(df_optimized_subset, df_planned_subset_from_session),
        'plot_investment_allocation_pie': create_investment_allocation_pie_chart(df_optimized_subset),
        'plot_cumulative_growth_curve': create_cumulative_growth_curve(df_planned_subset_from_session.copy()),
        'plot_roi_by_product': create_roi_by_product_chart(df_optimized_subset, df_planned_subset_from_session),
        'plot_volume_margin_tradeoff_bubble': create_volume_margin_tradeoff_bubble_chart(df_optimized_subset, df_planned_subset_from_session)
    }
    request.session.pop('session_error_message', None)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "total_subset_data": tf_subset_display_list, "total_subset_columns": tf_comparison_subset.columns.tolist(),
        "total_full_category_data": tf_full_cat_display_list, "total_full_category_columns": tf_full_cat_comp.columns.tolist(),
        "optimized_data": formatted_optimized_data_list, "optimized_columns": optimized_display_columns,
        "plots": plot_data, "session_id": session_id, "category_name": category_name,
        "num_optimized_products": len(df_optimized_subset),
        "optimization_goal": current_params_for_run['optimization_goal'], "processed": False,
        "session_error_message": None
    })


@app.post("/process_results_page", response_class=HTMLResponse)
async def process_results_page(request: Request, session_id: str = Form(...)):
    if session_id not in sessions:
        request.session['session_error_message'] = "Invalid session. Please start over."
        return RedirectResponse(f"/?error_message=Invalid session", status_code=303)

    current_app_session = sessions[session_id]
    if current_app_session.get('df_optimized_subset') is None:
        request.session['session_error_message'] = "Optimized data not found. Please run optimization."
        category_name = current_app_session.get('category_name')
        if category_name:
             return RedirectResponse(f"/go_to_product_selection?session_id={session_id}&category={category_name}&error_message=Optimized data missing", status_code=303)
        return RedirectResponse(f"/?session_id={session_id}&error_message=Optimized data missing", status_code=303)

    df_optimized_subset = current_app_session['df_optimized_subset']
    df_planned_subset_from_session = current_app_session['df_planned_subset']
    tf_final_comparison_subset = current_app_session.get('tf_final_comparison_subset')
    tf_full_category_comparison = current_app_session.get('tf_full_category_comparison')

    if tf_final_comparison_subset is None or tf_full_category_comparison is None:
        request.session['session_error_message'] = "Comparison data missing. Please re-run optimization."
        return RedirectResponse(f"/?session_id={session_id}&error_message=Comparison data missing", status_code=303)

    category_name = current_app_session['category_name']
    params_for_goal = current_app_session.get('params', {})
    optimization_goal = params_for_goal.get('optimization_goal', 'Revenue')

    tf_subset_display_list = []
    for _, rec_row in tf_final_comparison_subset.iterrows():
        rec = rec_row.to_dict(); fmt_rec = rec.copy()
        kind = "units" if rec['Data-Totals'] == 'Units' else "currency"
        fmt_rec['Planned'] = format_value(rec['Planned'], kind=kind); fmt_rec['Optimized'] = format_value(rec['Optimized'], kind=kind)
        if rec['% Change'] == 'inf': fmt_rec['% Change'] = "∞ %"
        elif pd.notna(rec['% Change']) and rec['% Change'] != '': fmt_rec['% Change'] = format_value(float(rec['% Change']), kind="percentage")
        else: fmt_rec['% Change'] = ''
        tf_subset_display_list.append(fmt_rec)

    tf_full_cat_display_list = []
    for _, rec_row_fc in tf_full_category_comparison.iterrows():
        rec_fc = rec_row_fc.to_dict(); fmt_rec_fc = rec_fc.copy()
        kind_fc = "units" if rec_fc['Data-Totals'] == 'Units' else "currency"
        fmt_rec_fc['Planned'] = format_value(rec_fc['Planned'], kind=kind_fc); fmt_rec_fc['Optimized State'] = format_value(rec_fc['Optimized State'], kind=kind_fc)
        if rec_fc['% Change'] == 'inf': fmt_rec_fc['% Change'] = "∞ %"
        elif pd.notna(rec_fc['% Change']) and rec_fc['% Change'] != '': fmt_rec_fc['% Change'] = format_value(float(rec_fc['% Change']), kind="percentage")
        else: fmt_rec_fc['% Change'] = ''
        tf_full_cat_display_list.append(fmt_rec_fc)

    df_planned_subset_for_results = df_planned_subset_from_session[['PL4 Sub-Category', 'Units', 'Revenue', 'Margin', 'Discount']].rename(
        columns={'Units': 'Planned Units', 'Revenue': 'Planned Revenue', 'Margin': 'Planned Margin', 'Discount': 'Planned Discount (%)'}
    )
    df_results_display_subset = pd.merge(df_optimized_subset, df_planned_subset_for_results, on='PL4 Sub-Category', how='left')
    optimized_display_columns = ['PL4 Sub-Category', 'Planned Units', 'Optimized Units', 'Planned Revenue', 'Optimized Revenue', 'Planned Margin', 'Optimized Margin', 'Planned Discount (%)', 'Optimized Discount', 'investment']
    for col in optimized_display_columns:
        if col not in df_results_display_subset.columns: df_results_display_subset[col] = 0 if 'Units' in col or 'Revenue' in col or 'Margin' in col or 'investment' in col else (0.0 if 'Discount' in col else "N/A")

    formatted_optimized_data_list = []
    for _, rec_row_opt in df_results_display_subset[optimized_display_columns].iterrows():
        rec_opt = rec_row_opt.to_dict(); new_rec = {k: rec_opt.get(k) for k in optimized_display_columns}
        new_rec['Planned Units'] = format_value(new_rec.get('Planned Units'), kind="units"); new_rec['Optimized Units'] = format_value(new_rec.get('Optimized Units'), kind="units")
        new_rec['Planned Revenue'] = format_value(new_rec.get('Planned Revenue'), kind="currency"); new_rec['Optimized Revenue'] = format_value(new_rec.get('Optimized Revenue'), kind="currency")
        new_rec['Planned Margin'] = format_value(new_rec.get('Planned Margin'), kind="currency"); new_rec['Optimized Margin'] = format_value(new_rec.get('Optimized Margin'), kind="currency")
        new_rec['Planned Discount (%)'] = format_value(new_rec.get('Planned Discount (%)',0.0), kind="percentage"); new_rec['Optimized Discount'] = format_value(new_rec.get('Optimized Discount',0.0), kind="percentage")
        new_rec['investment'] = format_value(new_rec.get('investment'), kind="currency")
        formatted_optimized_data_list.append(new_rec)

    plot_data = {
        'plot_planned_vs_optimized_margin': create_planned_vs_optimized_margin_plot(df_optimized_subset, df_planned_subset_from_session),
        'plot_planned_vs_optimized_revenue': create_planned_vs_optimized_revenue_plot(df_optimized_subset, df_planned_subset_from_session),
        **create_top_bottom_performers_charts(df_optimized_subset, df_planned_subset_from_session),
        'plot_investment_allocation_pie': create_investment_allocation_pie_chart(df_optimized_subset),
        'plot_cumulative_growth_curve': create_cumulative_growth_curve(df_planned_subset_from_session.copy()),
        'plot_roi_by_product': create_roi_by_product_chart(df_optimized_subset, df_planned_subset_from_session),
        'plot_volume_margin_tradeoff_bubble': create_volume_margin_tradeoff_bubble_chart(df_optimized_subset, df_planned_subset_from_session)
    }
    request.session.pop('session_error_message', None)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "total_subset_data": tf_subset_display_list, "total_subset_columns": tf_final_comparison_subset.columns.tolist(),
        "total_full_category_data": tf_full_cat_display_list, "total_full_category_columns": tf_full_category_comparison.columns.tolist(),
        "optimized_data": formatted_optimized_data_list, "optimized_columns": optimized_display_columns,
        "plots": plot_data, "processed": True, "session_id": session_id,
        "num_optimized_products": len(df_optimized_subset),
        "category_name": category_name, "optimization_goal": optimization_goal,
        "session_error_message": None
    })


if __name__ == "__main__":
    import uvicorn
    print(f"OPTIMIZER_VERBOSE_DEBUG is set to: {OPTIMIZER_VERBOSE_DEBUG}")
    if not os.path.exists("static"):
        os.makedirs("static")
    if not os.path.exists("templates"):
        os.makedirs("templates")
    # Ensure your HTML/CSS files are placed in templates/ and static/ respectively.
    uvicorn.run("main:app", host="0.0.0.0", port=8006, reload=True) # Using port 8006 as in original example