"""
Tab 3: Weekly Water Balance
The site weekly water balance update rebuilt from the actuals, with the model
forecast on a separate page (WeeklyReport.py).  Shown in the tab and downloaded
as a self-contained HTML file for printing.
"""

import streamlit as st
import streamlit.components.v1 as components

import CurrentReadings as cr
import WaterSupplyReport as wsr
import WeeklyReport as wr
from SiteActuals import SWB_PATH


def _mtime(path):
    return path.stat().st_mtime if path.exists() else 0


@st.cache_data(show_spinner='Building the weekly report and forecast...')
def weekly_html(week, swb_mtime, readings_mtime, config_mtime):
    """The mtimes refresh the cache when the workbook, the entered readings or the configuration change."""
    return wr.build_weekly_report(week)


def render_weekly_sidebar(historical_results):
    """Week selector: weeks with dam volumes and meter 014, latest first."""
    if historical_results is None:
        return
    weeks = wr.report_weeks(historical_results)[::-1]
    if not weeks:
        return
    st.sidebar.header("Weekly Report")
    st.sidebar.selectbox("Week ending", options=weeks, index=0, key="weekly_week",
                         format_func=lambda d: d.strftime('%d/%m/%Y'))


def render_weekly_tab(historical_results, reservoirs, pumps):
    st.header("Weekly Water Balance")
    if historical_results is None:
        st.warning("No site actuals available. Place Weekly SWB.xlsx in the project folder.")
        return
    weeks = wr.report_weeks(historical_results)
    if not weeks:
        st.warning("No week has both dam volumes and meter 014.")
        return
    week = st.session_state.get('weekly_week', weeks[-1])
    try:
        html = weekly_html(week, _mtime(SWB_PATH), _mtime(cr.READINGS_PATH), _mtime(wsr.CONFIG))
    except Exception as e:
        st.error(f"Weekly report error: {e}")
        st.exception(e)
        return
    st.caption("Page 1 is the site weekly update rebuilt from the actuals, by the site's method.  Page 2 is the "
               "model forecast.  Download the file and print it from a browser to A4 or PDF.")
    st.download_button("Download weekly report (HTML)", data=html, file_name=wr.report_filename(week),
                       mime="text/html", key="weekly_download")
    components.html(html, height=2350, scrolling=True)
