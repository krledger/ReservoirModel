"""
Plotting and visualization functions for reservoir system
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def detect_data_frequency(index):
    """Detect if data is daily, weekly, or other frequency based on index spacing"""
    if len(index) < 2:
        return 86400000  # Default to 1 day

    # Calculate median time difference between consecutive points
    diffs = [(index[i + 1] - index[i]).total_seconds() for i in range(min(10, len(index) - 1))]
    median_diff = np.median(diffs)

    # Determine frequency in milliseconds (make 1% larger to ensure no gaps)
    width_ms = int(median_diff * 1000 * 1.01)

    return width_ms


def create_plots(results, scenario_desc, start_year, end_year, r1_min, r2_min, river_pump_low, river_pump_high,
                 r1_turkeys, r2_surhs, results_prior_12m=None):
    """Create all plots for results visualisation
    
    Args:
        results: Current period results DataFrame
        scenario_desc: Description of scenario
        start_year: Start year for display
        end_year: End year for display
        r1_min: TND minimum capacity
        r2_min: SCD minimum capacity
        river_pump_low: River pump low cutoff
        river_pump_high: River pump high cutoff
        r1_turkeys: TND reservoir config
        r2_surhs: SCD reservoir config
        results_prior_12m: Optional DataFrame with results from 12 months prior for comparison
    """
    # Detect data frequency for proper bar width
    bar_width = detect_data_frequency(results.index)

    fig = make_subplots(
        rows=11, cols=1,
        subplot_titles=(
            'Surhs Creek Dam Reservoir Level (ML)',
            'Turkeys Nest Dam Reservoir Level (ML)',
            'Pump: River → TND (ML/day)',
            'Pump: TND → SCD (ML/day)',
            'Pump: SCD → Site (ML/day)',
            'River Flow (ML/day)',
            'Evaporation from Reservoirs (ML/day)',
            'Temperature - Ravenswood (°C)',
            'Relative Humidity - Ravenswood (%)',
            'Precipitation - Ravenswood (mm/day)',
            'Inflow to SCD - Surhs Creek (ML/day)'
        ),
        vertical_spacing=0.025,
        row_heights=[1.3, 1.3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        specs=[[{"secondary_y": False}]] * 10 + [[{"secondary_y": True}]]
    )

    # SCD (Surhs Creek) Level
    fig.add_trace(
        go.Scatter(x=results.index, y=results['r2_level_ML'],
                   name='Surhs Creek Dam', line=dict(color='teal', width=2),
                   fill='tozeroy', fillcolor='rgba(0,128,128,0.2)'),
        row=1, col=1
    )
    
    # Add prior 12 months average as horizontal line if available
    if results_prior_12m is not None and isinstance(results_prior_12m, dict):
        if 'r2_level_ML' in results_prior_12m and not pd.isna(results_prior_12m['r2_level_ML']):
            fig.add_hline(
                y=results_prior_12m['r2_level_ML'],
                line_dash="dash",
                line_color="teal",
                line_width=3,
                opacity=0.8,
                annotation_text=f"12mo avg: {results_prior_12m['r2_level_ML']:.0f}ML",
                annotation_position="right",
                row=1, col=1
            )
    
    fig.add_hline(y=r2_surhs['capacity_ML'], line_dash="dot", line_color="red",
                  annotation_text=f"Surhs Creek Capacity: {r2_surhs['capacity_ML']} ML", row=1, col=1)
    if r2_min > 0:
        fig.add_hline(y=r2_min, line_dash="dash", line_color="orange",
                      annotation_text=f"Surhs Creek Min: {r2_min} ML", row=1, col=1)

    # TND (Turkeys Nest) Level
    fig.add_trace(
        go.Scatter(x=results.index, y=results['r1_level_ML'],
                   name='Turkeys Nest Dam', line=dict(color='darkblue', width=2),
                   fill='tozeroy', fillcolor='rgba(0,0,139,0.2)'),
        row=2, col=1
    )
    
    # Add prior 12 months average as horizontal line if available
    if results_prior_12m is not None and isinstance(results_prior_12m, dict):
        if 'r1_level_ML' in results_prior_12m and not pd.isna(results_prior_12m['r1_level_ML']):
            fig.add_hline(
                y=results_prior_12m['r1_level_ML'],
                line_dash="dash",
                line_color="darkblue",
                opacity=0.6,
                annotation_text=f"12mo avg: {results_prior_12m['r1_level_ML']:.0f}ML",
                annotation_position="right",
                row=2, col=1
            )
    
    fig.add_hline(y=r1_turkeys['capacity_ML'], line_dash="dot", line_color="red",
                  annotation_text=f"Turkeys Nest Capacity: {r1_turkeys['capacity_ML']} ML", row=2, col=1)
    if r1_min > 0:
        fig.add_hline(y=r1_min, line_dash="dash", line_color="orange",
                      annotation_text=f"Turkeys Nest Min: {r1_min} ML", row=2, col=1)

    # Pump: River → TND
    if 'pump_river_to_r1_ML_day' in results.columns:
        fig.add_trace(
            go.Scatter(x=results.index, y=results['pump_river_to_r1_ML_day'],
                       name='River→TND Pump', line=dict(color='green', width=1.5),
                       fill='tozeroy', fillcolor='rgba(0,128,0,0.2)'),
            row=3, col=1
        )
        
        # Add prior 12 months average line if available
        if results_prior_12m is not None and isinstance(results_prior_12m, dict):
            if 'pump_river_to_r1_ML_day' in results_prior_12m and not pd.isna(results_prior_12m['pump_river_to_r1_ML_day']):
                fig.add_hline(
                    y=results_prior_12m['pump_river_to_r1_ML_day'],
                    line_dash="dash",
                    line_color="green",
                    line_width=2,
                    opacity=0.6,
                    annotation_text=f"12mo avg: {results_prior_12m['pump_river_to_r1_ML_day']:.1f}ML/day",
                    annotation_position="right",
                    row=3, col=1
                )

    # Pump: TND → SCD
    if 'pump_r1_to_r2_ML_day' in results.columns:
        fig.add_trace(
            go.Scatter(x=results.index, y=results['pump_r1_to_r2_ML_day'],
                       name='TND→SCD Pump', line=dict(color='blue', width=1.5),
                       fill='tozeroy', fillcolor='rgba(0,0,255,0.2)'),
            row=4, col=1
        )
        
        # Add prior 12 months average line if available
        if results_prior_12m is not None and isinstance(results_prior_12m, dict):
            if 'pump_r1_to_r2_ML_day' in results_prior_12m and not pd.isna(results_prior_12m['pump_r1_to_r2_ML_day']):
                fig.add_hline(
                    y=results_prior_12m['pump_r1_to_r2_ML_day'],
                    line_dash="dash",
                    line_color="blue",
                    line_width=2,
                    opacity=0.6,
                    annotation_text=f"12mo avg: {results_prior_12m['pump_r1_to_r2_ML_day']:.1f}ML/day",
                    annotation_position="right",
                    row=4, col=1
                )

    # Pump: SCD → Site
    if 'pump_r2_to_site_ML_day' in results.columns:
        fig.add_trace(
            go.Scatter(x=results.index, y=results['pump_r2_to_site_ML_day'],
                       name='SCD→Site Pump', line=dict(color='purple', width=1.5),
                       fill='tozeroy', fillcolor='rgba(128,0,128,0.2)'),
            row=5, col=1
        )
        
        # Add prior 12 months average line if available
        if results_prior_12m is not None and isinstance(results_prior_12m, dict):
            if 'pump_r2_to_site_ML_day' in results_prior_12m and not pd.isna(results_prior_12m['pump_r2_to_site_ML_day']):
                fig.add_hline(
                    y=results_prior_12m['pump_r2_to_site_ML_day'],
                    line_dash="dash",
                    line_color="purple",
                    line_width=2,
                    opacity=0.6,
                    annotation_text=f"12mo avg: {results_prior_12m['pump_r2_to_site_ML_day']:.1f}ML/day",
                    annotation_position="right",
                    row=5, col=1
                )

    # River Flow with Pumping Window
    if 'river_flow_ML_day' in results.columns and not results['river_flow_ML_day'].isna().all():
        pump_allowed = (results['river_flow_ML_day'] >= river_pump_low) & (
                results['river_flow_ML_day'] <= river_pump_high)
        pump_status_display = pump_allowed.astype(int) * 1000000

        fig.add_trace(
            go.Bar(x=results.index, y=pump_status_display,
                   name='Pumping Allowed', marker_color='green', opacity=0.8,
                   width=bar_width),
            row=6, col=1
        )
        fig.add_trace(
            go.Scatter(x=results.index, y=results['river_flow_ML_day'],
                       name='River Flow', line=dict(color='darkblue', width=1.5, shape='hv')),
            row=6, col=1
        )
        fig.add_hline(y=river_pump_low, line_dash="dash", line_color="grey", row=6, col=1)
        fig.add_hline(y=river_pump_high, line_dash="dash", line_color="grey", row=6, col=1)

    # Evaporation
    if 'r1_evap_ML_day' in results.columns and not results['r1_evap_ML_day'].isna().all():
        fig.add_trace(
            go.Scatter(x=results.index, y=results['r1_evap_ML_day'],
                       name='Evap - Turkeys Nest', line=dict(color='orange', width=1.5)),
            row=7, col=1
        )
    if 'r2_evap_ML_day' in results.columns and not results['r2_evap_ML_day'].isna().all():
        fig.add_trace(
            go.Scatter(x=results.index, y=results['r2_evap_ML_day'],
                       name='Evap - Surhs Creek', line=dict(color='darkorange', width=1.5)),
            row=7, col=1
        )

    # Temperature
    if 'temperature_degC' in results.columns and not results['temperature_degC'].isna().all():
        fig.add_trace(
            go.Scatter(x=results.index, y=results['temperature_degC'],
                       name='Temperature', line=dict(color='orangered', width=1.5)),
            row=8, col=1
        )

    # Humidity
    if 'relative_humidity_pct' in results.columns and not results['relative_humidity_pct'].isna().all():
        fig.add_trace(
            go.Scatter(x=results.index, y=results['relative_humidity_pct'],
                       name='Relative Humidity', line=dict(color='blue', width=1.5)),
            row=9, col=1
        )

    # Precipitation
    if 'precipitation_mm_day' in results.columns and not results['precipitation_mm_day'].isna().all():
        precip_normal = results['precipitation_mm_day'].copy()
        precip_normal[precip_normal > 30] = np.nan

        fig.add_trace(
            go.Bar(x=results.index, y=precip_normal,
                   name='Precipitation', marker_color='navy', opacity=0.6,
                   width=bar_width),
            row=10, col=1
        )

        precip_ma = results['precipitation_mm_day'].rolling(window=120, center=True).mean()
        fig.add_trace(
            go.Scatter(x=results.index, y=precip_ma,
                       name='120-day MA Precipitation',
                       line=dict(color='blue', width=2, dash='dash')),
            row=10, col=1
        )

        exceed_dates = results.index[results['precipitation_mm_day'] > 30]
        exceed_values = results['precipitation_mm_day'][results['precipitation_mm_day'] > 30]
        if len(exceed_dates) > 0:
            fig.add_trace(
                go.Scatter(x=exceed_dates, y=[30] * len(exceed_dates),
                           mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'),
                           name='Exceeds 30mm',
                           text=[f'{val:.1f}mm' for val in exceed_values]),
                row=10, col=1
            )

        fig.add_hline(y=r2_surhs['inflows']['fluvial']['min_rain_mm'],
                      line_dash="dash", line_color="orange", row=10, col=1)

    # Inflow to SCD
    if 'fluvial_inflow_ML' in results.columns and not results['fluvial_inflow_ML'].isna().all():
        fig.add_trace(
            go.Scatter(x=results.index, y=results['fluvial_inflow_ML'],
                       name='Fluvial Inflow', line=dict(color='darkgreen', width=1.5),
                       stackgroup='inflow'),
            row=11, col=1
        )
    if 'pluvial_inflow_r2_ML' in results.columns and not results['pluvial_inflow_r2_ML'].isna().all():
        fig.add_trace(
            go.Scatter(x=results.index, y=results['pluvial_inflow_r2_ML'],
                       name='Pluvial Inflow (Surhs Creek)', line=dict(color='lightgreen', width=1.5),
                       stackgroup='inflow'),
            row=11, col=1
        )

    if 'total_inflow_ML' in results.columns and not results['total_inflow_ML'].isna().all():
        total_inflow_ma = results['total_inflow_ML'].rolling(window=120, center=True).mean()
        fig.add_trace(
            go.Scatter(x=results.index, y=total_inflow_ma,
                       name='120-day MA Total Inflow',
                       line=dict(color='darkgreen', width=2, dash='dash')),
            row=11, col=1
        )

    # Update axes
    fig.update_xaxes(title_text="Date", row=11, col=1)
    fig.update_yaxes(title_text="ML", range=[0, r2_surhs['capacity_ML'] * 1.15], row=1, col=1)
    fig.update_yaxes(title_text="ML", range=[0, r1_turkeys['capacity_ML'] * 1.15], row=2, col=1)
    
    # Pump charts with dynamic ranges (add 20% headroom)
    if 'pump_river_to_r1_ML_day' in results.columns and not results['pump_river_to_r1_ML_day'].isna().all():
        max_val = results['pump_river_to_r1_ML_day'].max()
        fig.update_yaxes(title_text="ML/day", range=[0, max_val * 1.2], row=3, col=1)
    else:
        fig.update_yaxes(title_text="ML/day", row=3, col=1)
    
    if 'pump_r1_to_r2_ML_day' in results.columns and not results['pump_r1_to_r2_ML_day'].isna().all():
        max_val = results['pump_r1_to_r2_ML_day'].max()
        fig.update_yaxes(title_text="ML/day", range=[0, max_val * 1.2], row=4, col=1)
    else:
        fig.update_yaxes(title_text="ML/day", row=4, col=1)
    
    if 'pump_r2_to_site_ML_day' in results.columns and not results['pump_r2_to_site_ML_day'].isna().all():
        max_val = results['pump_r2_to_site_ML_day'].max()
        fig.update_yaxes(title_text="ML/day", range=[0, max_val * 1.2], row=5, col=1)
    else:
        fig.update_yaxes(title_text="ML/day", row=5, col=1)
    
    fig.update_yaxes(title_text="ML/day", type="log", range=[2, 6], row=6, col=1)
    fig.update_yaxes(title_text="ML/day", row=7, col=1)
    fig.update_yaxes(title_text="°C", row=8, col=1)
    fig.update_yaxes(title_text="%", row=9, col=1)
    fig.update_yaxes(title_text="mm", range=[0, 30], row=10, col=1)
    fig.update_yaxes(title_text="ML/day", range=[0, 5], row=11, col=1)

    fig.update_layout(
        height=5000,
        showlegend=True,
        title_text=f"Reservoir System Analysis - {scenario_desc} ({start_year}-{end_year})",
        hovermode='x unified',
        template='plotly_white',
        bargap=0,
        bargroupgap=0
    )

    return fig


def create_weekly_charts(chart_data, pumps, reservoirs, selected_week):
    """Create charts for weekly water balance report"""
    charts = {}
    
    # Chart 1: Burdekin Flow Upstream
    if 'river_flow_ML_day' in chart_data.columns and not chart_data['river_flow_ML_day'].isna().all():
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['river_flow_ML_day'],
            name='River Flow',
            line=dict(color='blue', width=2)
        ))

        fig1.add_hline(y=pumps[0]['cutoffs']['high_flow_ML_day'],
                       line_dash="dash", line_color="red",
                       annotation_text="Max pump flow")
        fig1.add_hline(y=pumps[0]['cutoffs']['low_flow_ML_day'],
                       line_dash="dash", line_color="orange",
                       annotation_text="Min pump flow")

        fig1.update_layout(
            height=300,
            yaxis_title="ML/day",
            yaxis_type="log",
            showlegend=True,
            template='plotly_white',
            margin=dict(l=50, r=50, t=30, b=50)
        )
        
        charts['river_flow'] = fig1

    # Chart 2: Dam Storage Volumes (%)
    fig2 = go.Figure()

    r1_pct = (chart_data['r1_level_ML'] / reservoirs[0]['capacity_ML']) * 100
    r2_pct = (chart_data['r2_level_ML'] / reservoirs[1]['capacity_ML']) * 100

    fig2.add_trace(go.Scatter(
        x=chart_data.index,
        y=r2_pct,
        name='SCD Level',
        line=dict(color='teal', width=2)
    ))

    fig2.add_trace(go.Scatter(
        x=chart_data.index,
        y=r1_pct,
        name='TND Level',
        line=dict(color='purple', width=2)
    ))

    fig2.add_hline(y=30, line_dash="dash", line_color="red",
                   annotation_text="min (30%)")

    fig2.update_layout(
        height=300,
        yaxis_title="%",
        yaxis_range=[0, 100],
        showlegend=True,
        template='plotly_white',
        margin=dict(l=50, r=50, t=30, b=50)
    )
    
    charts['storage_pct'] = fig2

    # Chart 3: Pump Flows
    fig3 = go.Figure()

    if 'pump_river_to_r1_ML_day' in chart_data.columns:
        fig3.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['pump_river_to_r1_ML_day'],
            name='River→TND',
            line=dict(color='green', width=2)
        ))

    if 'pump_r1_to_r2_ML_day' in chart_data.columns:
        fig3.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['pump_r1_to_r2_ML_day'],
            name='TND→SCD',
            line=dict(color='blue', width=2)
        ))

    if 'pump_r2_to_site_ML_day' in chart_data.columns:
        fig3.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['pump_r2_to_site_ML_day'],
            name='SCD→Site',
            line=dict(color='red', width=2)
        ))

    fig3.update_layout(
        height=300,
        yaxis_title="ML/day",
        showlegend=True,
        template='plotly_white',
        margin=dict(l=50, r=50, t=30, b=50)
    )
    
    charts['pump_flows'] = fig3
    
    return charts
