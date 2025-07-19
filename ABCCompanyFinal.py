nimport io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
#from dash import html, dcc


st.markdown(
    "<h1 style='text-align: left; font-size: 30px;'>ABC Company – Backlog Performance Dashboard</h1>",
    unsafe_allow_html=True
)

# Load data
df = pd.read_excel('Database test.xlsx', sheet_name='Data')
mapping = pd.read_excel('Database test.xlsx', sheet_name='mapping')

# Cleaning data
df = df.dropna()
df = df.drop_duplicates()
# df = df[df['code'].astype(str) != '0']
# df['code'] = df['code'].apply(lambda x: '000' if x == 0 else x)


df['code'] = df['code'].apply(lambda x: '000' if x == 0 else str(
    int(x)) if x.is_integer() else str(x))
df['BPC_CODE'] = df['BPC'].astype(str) + df['code']

df_merged = pd.merge(df, mapping, on='BPC_CODE', how='left')
df_merged = df_merged
df = df_merged.drop(df_merged.columns[[11, 12, 13, 14]], axis=1)


# Transform data
exclude_values = ['OE_TOT - OE Total', 'AM_TOT - Aftermarket Total']
df = df[~df['Partnership type'].isin(exclude_values)]

partnership_to_platform = {
    'OE_TOT_EXCL_IC - OE TOTAL EXCLUDING IC': 'CUST',
    'AM_TOT_EXCL_IC - AM TOTAL EXCLUDING IC': 'CUST',
    'OE.IC - OE Intercompany': 'IC',
    'AM.IC - AM Intercompany': 'IC',
    'ALLPRODUCTTYPES - All Product Types': 'Total'
}
df['Platform'] = df['Partnership type'].map(partnership_to_platform)

df['Date'] = pd.to_datetime('1899-12-30') + \
    pd.to_timedelta(df['Date'], unit='D')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()


# Sort by BPC and Date
df = df.sort_values(by=['Date', 'BPC', 'code', 'Partnership type'])

# Calculate MTD as the difference between the current and previous YTD values per BPC
df['actual_mtd'] = df.groupby(['BPC', 'code', 'Partnership type'])[
    'Actual past due backlog'].diff()

# The first value in the group will be NaN (since there is no previous value), so we replace it with the original value
mask = df['actual_mtd'].isna()
df.loc[mask, 'actual_mtd'] = df.loc[mask, 'Actual past due backlog']


df_2023 = df[df['Year'] == 2023].copy()
df_2023 = df_2023.sort_values(by=['BPC', 'Date'])

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
df_2023['Month_Name'] = pd.Categorical(
    df_2023['Month_Name'], categories=month_order, ordered=True)

monthly_bpc = df_2023.groupby(['Month_Name', 'BPC'], observed=False)[
    'actual_mtd'].sum().reset_index()
monthly_pt = df_2023.groupby(['Month_Name', 'Platform'], observed=False)[
    'actual_mtd'].sum().reset_index()

#  Create figure
fig = go.Figure()

#  Traces
for bpc in monthly_bpc['BPC'].unique():
    data = monthly_bpc[monthly_bpc['BPC'] == bpc]
    fig.add_trace(go.Scatter(x=data['Month_Name'], y=data['actual_mtd'],
                             mode='lines+markers',
                             name=f"BPC: {bpc}",
                             visible=True))

#  Traces
for pt in monthly_pt['Platform'].unique():
    data = monthly_pt[monthly_pt['Platform'] == pt]
    fig.add_trace(go.Scatter(x=data['Month_Name'], y=data['actual_mtd'],
                             name=f"Platform: {pt}",
                             visible=False))

#  Dropdown buttons
n_bpc = len(monthly_bpc['BPC'].unique())
n_pt = len(monthly_pt['Platform'].unique())

dropdown_buttons = [
    dict(label="Company Level",
         method="update",
         args=[{"visible": [True]*n_bpc + [False]*n_pt},
               {"title": {"text": " Monthly Actual Past Due Backlog by Company, 2023 "},
                "legend.title.text": "Company"}
               ]),

    dict(label="Platform Level",
         method="update",
         args=[{"visible": [False]*n_bpc + [True]*n_pt},
               {"title": {"text": " Monthly Actual Past Due Backlog by Platform, 2023"},
                "legend.title.text": "Platform"}
               ])
]
# Layout
fig.update_layout(
    template='plotly',
    updatemenus=[dict(
        type="dropdown",
        direction="down",
        x=1.1,
        y=1.15,
        showactive=True,
        active=0,
        buttons=dropdown_buttons
    )],
    title="Monthly Actual Past Due Backlog (MTD) by Company and by Platform, 2023",
    xaxis_title="Month",
    yaxis_title="Actual Past Due Backlog (USD)",
    legend_title="Company",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig, use_container_width=True)


df['Year-Month'] = df['Date'].dt.to_period('M').astype(str)

# Filter last 3 years
last_3_years = df['Year'].max() - 3
df_3y = df[df['Year'] >= last_3_years].copy()

# Grouping by Month
monthly_trend = df_3y.groupby(
    'Year-Month')['Actual past due backlog'].sum().reset_index()

# Checking
# print(monthly_trend.shape)
# print(monthly_trend.head())

# Conversion
monthly_trend['Actual past due backlog'] = pd.to_numeric(
    monthly_trend['Actual past due backlog'], errors='coerce')

# Sort by Month
monthly_trend = monthly_trend.sort_values(by='Year-Month')

fig = px.line(monthly_trend,
              x='Year-Month',
              y='Actual past due backlog',
              title='Actual Past Due Backlog Trend - Last 3 Years',
              markers=True,
              labels={'Year-Month': 'Month', 'Actual past due backlog': 'Past Due Backlog'})

fig.update_traces(line=dict(color='#636EFA'), marker=dict(color='#636EFA'))

fig.update_layout(
    xaxis_tickangle=-45,
    xaxis_title='Month',
    yaxis_title='Past Due Backlog (USD)',
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

#

bu_df = mapping[mapping['CLASS'] == 'BU'].copy()
site_df = mapping[mapping['CLASS'] == 'SITES'].copy()

df['code'] = df['code'].astype(int).astype(str).str.zfill(4)
mapping['code_clean'] = mapping['BPC_CODE'].astype(str).str[-4:]

code_to_region = mapping.set_index('code_clean')['Region_Group'].to_dict()
df['Region_Group'] = df['code'].map(code_to_region)

site_backlog = df.groupby('Region_Group')[
    'Actual past due backlog'].sum().reset_index()
site_backlog = site_backlog.sort_values(
    by='Actual past due backlog', ascending=False)

site_backlog = site_backlog[site_backlog['Region_Group'] != 'all']
filtered_backlog = site_backlog[site_backlog['Region_Group'] != 'all'].head(10)

fig = px.bar(
    filtered_backlog,
    x='Actual past due backlog',
    y='Region_Group',
    color='Region_Group',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    orientation='h',
    title='Worst 10 Product Types by Past Due Backlog',
    labels={
        'Region_Group': '',
        'Actual past due backlog': 'Backlog (USD)'
    }
)
st.plotly_chart(fig, use_container_width=True)
 
top3 = filtered_backlog.head(3).copy()

top3['Backlog (formatted)'] = top3['Actual past due backlog'].apply(
    lambda x: f"${x / 1e9:.1f} billion"
)

top3_table = top3[['Region_Group', 'Backlog (formatted)']]
top3_table.columns = ['3 Worst Product Type', 'Past Due Backlog']
top3_table.index = range(1, len(top3_table) + 1)
st.write("3 Worst Product Types ( Highest Past Due Backlog):")
st.dataframe(top3_table)

# Total backlog
total_backlog = filtered_backlog['Actual past due backlog'].sum()

# Backlog for top3
top3_backlog = top3['Actual past due backlog'].sum()

# % Proportion of the total
top3_share_percent = (top3_backlog / total_backlog) * 100

st.write(
    f"Top 3 product types contribute {top3_share_percent:.2f}% of the total past due backlog.")

other_backlog = filtered_backlog.iloc[3:]['Actual past due backlog'].sum()
top3.loc[len(top3.index)] = {
    'Region_Group': 'others',
    'Actual past due backlog': other_backlog
}

# Pie chart
fig = px.pie(top3,
             values='Actual past due backlog',
             names='Region_Group',
             title='Contribution to Total Past Due Backlog: Worst 3 vs Others',
             hole=0,
             color_discrete_sequence=px.colors.qualitative.Plotly)

fig.update_traces(textinfo='percent+label', textposition='outside')

st.plotly_chart(fig, use_container_width=True)

df_ic_cust = df[df['Platform'].isin(['IC', 'CUST'])].copy()

# Filter last 3 years
last_3_years = df_ic_cust['Year'].max() - 2
df_ic_cust = df_ic_cust[df_ic_cust['Year'] >= last_3_years]

monthly_trend = df_ic_cust.groupby(
    ['Year-Month', 'Platform'])['Actual past due backlog'].sum().reset_index()

fig = px.line(
    monthly_trend,
    x='Year-Month',
    y='Actual past due backlog',
    color='Platform',
    markers=True,
    title='Past Due Backlog Trend – IC vs CUST (Last 3 Years)',
    labels={
        'Year-Month': 'Month',
        'Actual past due backlog': 'Past Due Backlog (USD)',
        'Platform': 'Type'
    },
    color_discrete_sequence=px.colors.qualitative.Plotly
)


fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Actual Past Due Backlog (USD)",
    legend_title="Platform",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, use_container_width=True)


df = df[df['PRODUCT GROUP'].notna()]
# num_rows_with_nan = df_new.isna().any(axis=1).sum()
# st.write(f"Number of rows with at least one NaN: {num_rows_with_nan}")
# Create a 'Year-Month' column for monthly grouping
df['Year-Month'] = df['Date'].dt.to_period('M').astype(str)

df = df[df['Region_Group'] != 'all']

# Calculate total backlog per month
total_backlog = df.groupby(
    'Year-Month')['Actual past due backlog'].sum().reset_index()
total_backlog = total_backlog.rename(
    columns={'Actual past due backlog': 'Total Past Due Backlog'})

# Sidebar dropdown to select the hierarchy level
level = st.sidebar.selectbox(
    "Select level for the graph Past Due Backlog % Trend:",
    ['PRODUCT GROUP', 'Region_Group', 'SITE_NAME'],
    key='level_selectbox'
)
# Group data by selected level and month
grouped = df.groupby(['Year-Month', level]
                     )['Actual past due backlog'].sum().reset_index()

# Merge with total backlog to calculate percentage share
merged = grouped.merge(total_backlog, on='Year-Month')
merged['Past Due Backlog %'] = 100 * \
    merged['Actual past due backlog'] / merged['Total Past Due Backlog']
merged = merged.groupby(['Year-Month', level]
                        )['Past Due Backlog %'].sum().reset_index()


# Map for nicer titles in the chart
title_map = {
    'PRODUCT GROUP': 'Product Group',
    'Region_Group': 'Business Unit (Region)',
    'SITE_NAME': 'Site'
}


title = f"Past Due Backlog % Trend by {title_map[level]}"

fig = px.bar(
    merged,
    x='Year-Month',
    y='Past Due Backlog %',
    color=level,
    title=f"Past Due Backlog % Trend by {title_map[level]}",
    labels={
        'Year-Month': 'Month',
        'Past Due Backlog %': 'Past Due Backlog (%)',
        level: title_map[level]
    },
    barmode='stack'
)

fig.update_layout(
    xaxis_tickangle=-45,
    hovermode='x unified',
    height=500
)

# fig.update_traces(
#   hovertemplate=f"{title_map[level]}<br>" +
#                 "Date: %{x}<br>" +
#                "Past Due Backlog: %{y:.1f}%"
# )

fig.for_each_trace(lambda t: t.update(
    hovertemplate='<b>Product Group : ' + t.name + '</b><br>' +
                  'Date: %{x}<br>' +
                  'Past Due Backlog: %{y:.1f}%'
))

fig.update_layout(
    barmode='stack',
    title=title,
    xaxis_title='Date',
    yaxis_title='Past Due Backlog (%)',
    xaxis_tickangle=-45,
    hovermode='x unified',
    height=500
)

fig.update_traces(name="", selector=dict(type='bar'))

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)


st.write("Preview of df_new DataFrame:")
st.dataframe(df.head(20))

output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1')
output.seek(0)

st.download_button(
    label="Download Excel file",
    data=output,
    file_name="df_new_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# output = io.BytesIO()
# with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#   df_new.to_excel(writer, index=False, sheet_name='Sheet1')
#   writer.save()
# output.seek(0)

# st.download_button(
#   label="Download Excel file",
#   data=output,
#   file_name="df_new_data.xlsx",
#  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# )
site_impact = df.groupby('SITE_NAME')[
    'Actual past due backlog'].mean().reset_index()

# Sort and choose
top_sites = site_impact.sort_values(
    by='Actual past due backlog', ascending=False).head(10)

fig = px.bar(
    top_sites,
    x='Actual past due backlog',
    y='SITE_NAME',
    orientation='h',
    title='Worst 10 Sites by Impact on Past Due Backlog',
    labels={
        'Actual past due backlog': 'Past Due Backlog (USD)', 'SITE_NAME': 'Site'},
    color='Actual past due backlog',
    color_continuous_scale=px.colors.sequential.Viridis,
    text=None
)

fig.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    coloraxis_colorbar=dict(title="Backlog (USD)")
)

st.plotly_chart(fig, use_container_width=True)


top3_sites = top_sites.head(3).copy()

top3_sites['Backlog (formatted)'] = top3_sites['Actual past due backlog'].apply(
    lambda x: f"${x / 1e9:.1f} billion"
) 

top3_sites=top3_sites.merge(mapping, on='SITE_NAME', how='left')

top3_sites_table = top3_sites[['SITE_NAME', 'Backlog (formatted)','Region_Group']]
top3_sites_table.columns = ['3 Worst Sites', 'Past Due Backlog','Business unit']
top3_sites_table.index = range(1, len(top3_table) + 1)


st.write("3 Worst Sites ( Highest Past Due Backlog):")
st.dataframe(top3_sites_table)

top3_sites.loc[len(top3_sites.index)] = {
    'SITE_NAME': 'others',
    'Actual past due backlog': other_backlog
}

# Pie chart
fig = px.pie(top3_sites,
             values='Actual past due backlog',
             names='SITE_NAME',
             title='Contribution to Total Past Due Backlog: Worst 3 vs Others',
             hole=0,
             color_discrete_sequence=px.colors.qualitative.Plotly)

fig.update_traces(textinfo='percent+label', textposition='outside')

st.plotly_chart(fig, use_container_width=True)

bottom3_sites = top_sites.tail(3).copy()

bottom3_sites['Backlog (formatted)'] = bottom3_sites['Actual past due backlog'].apply(
    lambda x: f"${x / 1e9:.1f} billion"
)

bottom3_sites=bottom3_sites.merge(mapping, on='SITE_NAME', how='left')

bottom3_sites_table = bottom3_sites[['SITE_NAME', 'Backlog (formatted)','Region_Group']]
bottom3_sites_table.columns = ['3 Best Sites', 'Past Due Backlog', 'Business unit']
bottom3_sites_table.index = range(1, len(top3_table) + 1)
st.write("3 Best Sites (Lowest Past Due Backlog):")
st.dataframe(bottom3_sites_table)
