"""
Tab 3: Weekly Water Balance Report
Handles weekly reporting and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from reservoir_plotting import create_weekly_charts


def render_weekly_tab(historical_results, reservoirs, pumps):
    """Render the weekly water balance tab"""
    
    st.header("Weekly Water Balance Report")

    if historical_results is not None:
        try:
            # Aggregate to weekly data
            weekly_data = aggregate_to_weekly(historical_results)
            
            if len(weekly_data) < 2:
                st.warning("Need at least 2 weeks of data for weekly reports")
            else:
                # Week selector from sidebar
                selected_week_idx = st.session_state.get('week_selector', len(weekly_data) - 1)
                
                if selected_week_idx >= len(weekly_data):
                    selected_week_idx = len(weekly_data) - 1
                
                selected_week = weekly_data.index[selected_week_idx]
                current_week = weekly_data.iloc[selected_week_idx]

                # Previous week for comparison
                if selected_week_idx > 0:
                    previous_week = weekly_data.iloc[selected_week_idx - 1]
                    has_previous = True
                else:
                    previous_week = None
                    has_previous = False

                # Display weekly report
                display_weekly_header(selected_week)
                
                # Main layout
                left_col, right_col = st.columns([1, 2])
                
                with left_col:
                    display_weekly_metrics(current_week, previous_week, has_previous, reservoirs, pumps)
                
                with right_col:
                    display_weekly_charts(historical_results, selected_week, pumps, reservoirs)

        except Exception as e:
            st.error(f"Error displaying weekly report: {e}")
            st.exception(e)
    else:
        st.warning("No historical data available for weekly reports")
def aggregate_to_weekly(historical_results):
    """Aggregate historical data to weekly periods"""
    # Build aggregation dictionary only for columns that exist
    agg_dict = {
        'r1_level_ML': 'last',
        'r2_level_ML': 'last',
        'pump_river_to_r1_ML_day': 'mean',
        'pump_r1_to_r2_ML_day': 'mean',
        'pump_r2_to_site_ML_day': 'mean'
    }

    # Only add river_flow_ML_day if it exists
    if 'river_flow_ML_day' in historical_results.columns:
        agg_dict['river_flow_ML_day'] = 'mean'

    # Resample to weekly (week ending Sunday)
    weekly_data = historical_results.resample('W-SUN').agg(agg_dict)

    # Filter out weeks with no data
    weekly_data = weekly_data.dropna(subset=['r1_level_ML', 'r2_level_ML'])
    
    return weekly_data


def display_weekly_header(selected_week):
    """Display header for weekly report"""
    month_name = selected_week.strftime('%b')
    week_num = (selected_week.day - 1) // 7 + 1

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("## SITE WATER BALANCE - WEEKLY UPDATE")
    with col2:
        st.markdown(f"### {month_name}")
    with col3:
        st.markdown(f"### WEEK {week_num}")

    st.markdown("---")


def display_weekly_metrics(current_week, previous_week, has_previous, reservoirs, pumps):
    """Display metrics for weekly report"""
    
    # Available water at current demand
    st.markdown("### AVAILABLE WATER AT CURRENT DEMAND")
    
    total_storage = current_week['r1_level_ML'] + current_week['r2_level_ML']
    daily_demand = current_week['pump_r2_to_site_ML_day']
    
    # Estimate losses
    r1_seepage = reservoirs[0]['losses']['seepage_ML_day']
    r2_seepage = reservoirs[1]['losses']['seepage_ML_day']
    total_losses = r1_seepage + r2_seepage + 2.0  # 2 ML/day evaporation estimate
    
    days_storage = total_storage / (daily_demand + total_losses) if (daily_demand + total_losses) > 0 else 0
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("STORAGE VOLUME (TND + SCD) ML", f"{int(total_storage)}")
    with metric_col2:
        st.metric("DAYS STORAGE AT CURRENT USE", f"{int(days_storage)}")
    
    st.markdown("---")
    
    # Burdekin River
    st.markdown("### BURDEKIN RIVER")
    
    has_river_flow = 'river_flow_ML_day' in current_week.index
    
    if has_river_flow:
        river_flow_current = current_week['river_flow_ML_day']
        river_flow_last = previous_week['river_flow_ML_day'] if has_previous else None
        
        st.markdown("**DAILY RIVER FLOW (ML)**")
        if not pd.isna(river_flow_current):
            delta_flow = river_flow_current - river_flow_last if has_previous and river_flow_last else None
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current", f"{int(river_flow_current)}",
                         delta=f"{int(delta_flow)}" if delta_flow else None)
            with col2:
                st.metric("Last Week", f"{int(river_flow_last)}" if river_flow_last else "N/A")
        else:
            st.info("No river flow data available")
    st.markdown("---")
    
    # Dam Storage
    st.markdown("### DAM STORAGE")
    
    r1_current_ml = current_week['r1_level_ML']
    r1_current_pct = (r1_current_ml / reservoirs[0]['capacity_ML']) * 100
    r1_last_ml = previous_week['r1_level_ML'] if has_previous else None
    r1_last_pct = (r1_last_ml / reservoirs[0]['capacity_ML']) * 100 if r1_last_ml else None
    
    st.markdown("**TND STORAGE**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("%", f"{int(r1_current_pct)}",
                 delta=f"{int(r1_current_pct - r1_last_pct)}" if r1_last_pct else None)
    with col2:
        st.metric("ML (Current)", f"{int(r1_current_ml)}")
    with col3:
        st.metric("ML (Last)", f"{int(r1_last_ml)}" if r1_last_ml else "N/A")
    
    r2_current_ml = current_week['r2_level_ML']
    r2_current_pct = (r2_current_ml / reservoirs[1]['capacity_ML']) * 100
    r2_last_ml = previous_week['r2_level_ML'] if has_previous else None
    r2_last_pct = (r2_last_ml / reservoirs[1]['capacity_ML']) * 100 if r2_last_ml else None
    
    st.markdown("**SCD STORAGE**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("%", f"{int(r2_current_pct)}",
                 delta=f"{int(r2_current_pct - r2_last_pct)}" if r2_last_pct else None)
    with col2:
        st.metric("ML (Current)", f"{int(r2_current_ml)}")
    with col3:
        st.metric("ML (Last)", f"{int(r2_last_ml)}" if r2_last_ml else "N/A")


def display_weekly_charts(historical_results, selected_week, pumps, reservoirs):
    """Display charts for weekly report"""
    st.markdown("#### Charts - Last 180 Days")
    
    # Get data for chart period
    chart_data = historical_results[historical_results.index <= selected_week].tail(180)
    
    if len(chart_data) > 0:
        charts = create_weekly_charts(chart_data, pumps, reservoirs, selected_week)
        
        if 'river_flow' in charts:
            st.markdown("#### Burdekin flow upstream of TND")
            st.plotly_chart(charts['river_flow'], use_container_width=True)
        
        if 'storage_pct' in charts:
            st.markdown("#### Dam Storage Volumes (%)")
            st.plotly_chart(charts['storage_pct'], use_container_width=True)
        
        if 'pump_flows' in charts:
            st.markdown("#### Pump Flows (ML/day)")
            st.plotly_chart(charts['pump_flows'], use_container_width=True)
        
        # Placeholders for additional charts
        st.markdown("#### Recycled Water - Flows (ML/day)")
        st.info("Chart placeholder - data not yet available")
        
        st.markdown("#### Raw Water Use - Flows (ML/day)")
        st.info("Chart placeholder - data not yet available")
    else:
        st.warning("No data available for charts")

def render_weekly_sidebar(historical_results):
    """Render sidebar controls for weekly tab"""
    if historical_results is not None:
        weekly_data = aggregate_to_weekly(historical_results)
        
        if len(weekly_data) >= 2:
            week_dates = weekly_data.index.tolist()
            week_labels = [f"{d.strftime('%d %b %Y')} (Week ending)" for d in week_dates]

            st.sidebar.header("Weekly Report Controls")
            selected_week_idx = st.sidebar.selectbox(
                "Select Week Ending",
                options=range(len(week_labels)),
                format_func=lambda i: week_labels[i],
                index=len(week_labels) - 1,
                key="week_selector"
            )
