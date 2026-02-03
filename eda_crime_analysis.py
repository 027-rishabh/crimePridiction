#!/usr/bin/env python3
"""
Exploratory Data Analysis for Crime Prediction Dataset
Generates insights and visualizations for the crime data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')  # Using a standard style that's widely available

def load_data():
    """Load the cleaned data"""
    print("Loading data...")
    
    # Load the cleaned data
    cleaned_path = "data/cleaned/crime_data_cleaned.csv"
    features_path = "data/features/crime_data_features.csv"
    
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        print(f"Loaded {len(df)} rows from {cleaned_path}")
    elif os.path.exists(features_path):
        df = pd.read_csv(features_path)
        print(f"Loaded {len(df)} rows from {features_path}")
    else:
        # Look for raw data files
        raw_files = [
            "data/raw/districtwise-crime-against-children-2017-onwards.csv",
            "data/raw/districtwise-crime-against-scs-2017-onwards.csv", 
            "data/raw/districtwise-crime-against-sts-2017-onwards.csv",
            "data/raw/districtwise-crime-against-women-2017-onwards.csv"
        ]
        
        dfs = []
        for file_path in raw_files:
            if os.path.exists(file_path):
                df_part = pd.read_csv(file_path)
                # Add a column to identify the protected group
                if 'children' in file_path.lower():
                    df_part['protected_group'] = 'Children'
                elif 'scs' in file_path.lower():
                    df_part['protected_group'] = 'SC'
                elif 'sts' in file_path.lower():
                    df_part['protected_group'] = 'ST'
                elif 'women' in file_path.lower():
                    df_part['protected_group'] = 'Women'
                dfs.append(df_part)
                print(f"Loaded {len(df_part)} rows from {file_path}")
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            print(f"Combined data: {len(df)} rows")
        else:
            raise FileNotFoundError("No data files found in expected locations")
    
    return df

def basic_info(df):
    """Print basic information about the dataset"""
    print("\n" + "="*50)
    print("BASIC DATASET INFORMATION")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for protected group column
    if 'protected_group' in df.columns:
        print(f"Protected groups: {df['protected_group'].value_counts().to_dict()}")
    
    print(f"Years range: {df['year'].min()} - {df['year'].max()}")
    print(f"States: {df['state_name'].nunique()}")
    print(f"Districts: {df['district_name'].nunique()}")
    
    # Missing values
    missing_vals = df.isnull().sum()
    missing_percent = (missing_vals / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_vals, 'Percentage': missing_percent})
    missing_df = missing_df[missing_df['Percentage'] > 0].sort_values('Percentage', ascending=False)
    
    if not missing_df.empty:
        print(f"\nTop 5 columns with missing values:")
        print(missing_df.head())
    
    return missing_df

def analyze_crime_trends(df):
    """Analyze crime trends over time"""
    print("\n" + "="*50)
    print("CRIME TRENDS ANALYSIS")
    print("="*50)
    
    # Group by year and protected group
    yearly_trends = df.groupby(['year', 'protected_group'])['total_crimes'].sum().reset_index()
    yearly_stats = df.groupby(['year', 'protected_group']).agg({
        'total_crimes': ['count', 'mean', 'sum'],
        'violent_crimes': 'sum',
        'sexual_crimes': 'sum',
        'property_crimes': 'sum',
        'kidnapping_crimes': 'sum'
    }).reset_index()
    
    # Flatten column names
    yearly_stats.columns = ['year', 'protected_group', 'count_records', 'avg_total_crimes', 'sum_total_crimes',
                          'sum_violent_crimes', 'sum_sexual_crimes', 'sum_property_crimes', 'sum_kidnapping_crimes']
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total crimes over time by group
    if 'protected_group' in df.columns:
        sns.lineplot(data=yearly_trends, x='year', y='total_crimes', hue='protected_group', ax=axes[0,0])
        axes[0,0].set_title('Total Crimes Over Time by Protected Group')
        axes[0,0].ticklabel_format(style='plain', axis='y')
    else:
        # If no protected group, just show overall trends
        yearly_agg = df.groupby('year')['total_crimes'].sum().reset_index()
        sns.lineplot(data=yearly_agg, x='year', y='total_crimes', ax=axes[0,0])
        axes[0,0].set_title('Total Crimes Over Time')
    
    # 2. Crime types over time
    if all(col in df.columns for col in ['year', 'violent_crimes', 'sexual_crimes', 'property_crimes']):
        crime_types = df.groupby('year')[['violent_crimes', 'sexual_crimes', 'property_crimes']].sum().reset_index()
        crime_melted = crime_types.melt(id_vars=['year'], 
                                       value_vars=['violent_crimes', 'sexual_crimes', 'property_crimes'],
                                       var_name='crime_type', value_name='count')
        sns.lineplot(data=crime_melted, x='year', y='count', hue='crime_type', ax=axes[0,1])
        axes[0,1].set_title('Different Types of Crimes Over Time')
        axes[0,1].ticklabel_format(style='plain', axis='y')
    
    # 3. Top 10 states by total crimes
    if 'state_name' in df.columns:
        state_crimes = df.groupby('state_name')['total_crimes'].sum().sort_values(ascending=False).head(10)
        sns.barplot(x=state_crimes.values, y=state_crimes.index, ax=axes[1,0])
        axes[1,0].set_title('Top 10 States by Total Crimes')
        axes[1,0].set_xlabel('Total Crimes')
    
    # 4. Distribution of total crimes
    if 'total_crimes' in df.columns:
        axes[1,1].hist(df['total_crimes'].dropna(), bins=50, edgecolor='black')
        axes[1,1].set_title('Distribution of Total Crimes')
        axes[1,1].set_xlabel('Total Crimes')
        axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('eda_crime_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return yearly_trends, yearly_stats

def analyze_protected_groups(df):
    """Analyze differences across protected groups"""
    print("\n" + "="*50)
    print("PROTECTED GROUPS ANALYSIS")
    print("="*50)
    
    if 'protected_group' not in df.columns:
        print("No protected_group column found - skipping group analysis")
        return
    
    # Group statistics
    group_stats = df.groupby('protected_group').agg({
        'total_crimes': ['count', 'mean', 'std', 'min', 'max'],
        'violent_crimes': ['mean', 'sum'],
        'sexual_crimes': ['mean', 'sum'],
        'property_crimes': ['mean', 'sum']
    }).round(2)
    
    print("Statistics by Protected Group:")
    print(group_stats)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Average crimes by group
    avg_crimes_by_group = df.groupby('protected_group')['total_crimes'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_crimes_by_group.values, y=avg_crimes_by_group.index, ax=axes[0,0])
    axes[0,0].set_title('Average Total Crimes by Protected Group')
    axes[0,0].set_xlabel('Average Total Crimes')
    
    # 2. Crime distribution by group
    if 'year' in df.columns:
        # Average crimes per year by group
        avg_by_group_year = df.groupby(['protected_group', 'year'])['total_crimes'].mean().reset_index()
        sns.lineplot(data=avg_by_group_year, x='year', y='total_crimes', hue='protected_group', ax=axes[0,1])
        axes[0,1].set_title('Average Crimes by Year and Protected Group')
    
    # 3. Violent vs Sexual crimes by group
    if 'violent_crimes' in df.columns and 'sexual_crimes' in df.columns:
        group_violent_sexual = df.groupby('protected_group')[['violent_crimes', 'sexual_crimes']].mean()
        group_violent_sexual.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Average Violent vs Sexual Crimes by Group')
        axes[1,0].set_xlabel('Protected Group')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Box plot of total crimes by group
    sns.boxplot(data=df, x='total_crimes', y='protected_group', ax=axes[1,1])
    axes[1,1].set_title('Distribution of Total Crimes by Protected Group')
    axes[1,1].set_xlabel('Total Crimes')
    
    plt.tight_layout()
    plt.savefig('eda_protected_groups.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return group_stats

def analyze_correlations(df):
    """Analyze correlations between features"""
    print("\n" + "="*50)
    print("FEATURE CORRELATIONS ANALYSIS")
    print("="*50)
    
    # Select numerical features for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Focus on key crime-related features
    crime_cols = ['total_crimes', 'violent_crimes', 'sexual_crimes', 'property_crimes', 'kidnapping_crimes']
    feature_cols = [col for col in numeric_cols if col not in ['id', 'year', 'state_code', 'district_code'] and 
                   (col in crime_cols or 'crime' in col.lower() or 'rape' in col.lower() or 
                    'assault' in col.lower() or 'kidnap' in col.lower() or 'murder' in col.lower())]
    
    if len(feature_cols) < 2:
        # If limited crime columns, expand to more general features
        feature_cols = [col for col in numeric_cols if col not in ['id'] and df[col].nunique() > 1]
        feature_cols = feature_cols[:min(20, len(feature_cols))]  # Limit to 20 features for readability
    
    print(f"Analyzing correlations for {len(feature_cols)} features")
    
    if len(feature_cols) > 1:
        corr_matrix = df[feature_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Crime Features')
        plt.tight_layout()
        plt.savefig('eda_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], abs(corr_val)))
        
        corr_pairs.sort(key=lambda x: x[2], reverse=True)
        print("\nTop 10 correlated feature pairs (absolute correlation):")
        for i, (feat1, feat2, corr) in enumerate(corr_pairs[:10]):
            print(f"{i+1:2d}. {feat1} vs {feat2}: {corr:.3f}")
        
        return corr_matrix
    else:
        print("Not enough numerical features for correlation analysis")
        return None

def analyze_states_districts(df):
    """Analyze crime patterns across states and districts"""
    print("\n" + "="*50)
    print("STATE AND DISTRICT ANALYSIS")
    print("="*50)
    
    if 'state_name' not in df.columns or 'district_name' not in df.columns:
        print("State or district columns not found - skipping analysis")
        return
    
    # State analysis
    state_analysis = df.groupby('state_name').agg({
        'total_crimes': ['count', 'sum', 'mean'],
        'year': 'nunique'  # Number of years covered
    }).round(2)
    
    state_analysis.columns = ['crime_records', 'total_crimes', 'avg_crimes_per_record', 'years_covered']
    state_analysis = state_analysis.sort_values('total_crimes', ascending=False)
    
    print("Top 10 States by Total Crimes:")
    print(state_analysis.head(10))
    
    # District analysis
    district_analysis = df.groupby('district_name').agg({
        'total_crimes': ['count', 'sum', 'mean']
    }).round(2)
    
    district_analysis.columns = ['crime_records', 'total_crimes', 'avg_crimes_per_record']
    district_analysis = district_analysis.sort_values('total_crimes', ascending=False)
    
    print(f"\nTop 10 Districts by Total Crimes:")
    print(district_analysis.head(10))
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top 10 states by total crimes
    top_states = state_analysis.head(10)['total_crimes']
    sns.barplot(x=top_states.values, y=top_states.index, ax=axes[0,0])
    axes[0,0].set_title('Top 10 States by Total Crimes')
    axes[0,0].set_xlabel('Total Crimes')
    
    # 2. Years of coverage by state
    state_coverage = df.groupby('state_name')['year'].nunique().sort_values(ascending=False)
    sns.barplot(x=state_coverage.values, y=state_coverage.index, ax=axes[0,1])
    axes[0,1].set_title('Years of Data Coverage by State')
    axes[0,1].set_xlabel('Number of Years')
    
    # 3. Top 10 districts by total crimes
    top_districts = district_analysis.head(10)['total_crimes']
    sns.barplot(x=top_districts.values, y=top_districts.index, ax=axes[1,0])
    axes[1,0].set_title('Top 10 Districts by Total Crimes')
    axes[1,0].set_xlabel('Total Crimes')
    
    # 4. Average crimes per state
    state_avg = df.groupby('state_name')['total_crimes'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=state_avg.values, y=state_avg.index, ax=axes[1,1])
    axes[1,1].set_title('Top 10 States by Average Crimes per Record')
    axes[1,1].set_xlabel('Average Crimes')
    
    plt.tight_layout()
    plt.savefig('eda_states_districts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return state_analysis, district_analysis

def generate_eda_dashboard(df):
    """Generate a comprehensive EDA dashboard with Plotly"""
    print("\n" + "="*50)
    print("GENERATING INTERACTIVE DASHBOARD")
    print("="*50)
    
    # Create interactive plots with plotly
    # Crime trends over time
    if 'year' in df.columns and 'total_crimes' in df.columns:
        yearly_crime = df.groupby('year')['total_crimes'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_crime['year'],
            y=yearly_crime['total_crimes'],
            mode='lines+markers',
            name='Total Crimes',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title='Total Crimes Over Time',
            xaxis_title='Year',
            yaxis_title='Total Crimes',
            hovermode='x unified'
        )
        
        fig.write_html('dashboard_crime_trends.html')
        print("Saved crime trends dashboard: dashboard_crime_trends.html")
    
    # Crime by protected group (if available)
    if 'protected_group' in df.columns and 'total_crimes' in df.columns:
        group_crime = df.groupby('protected_group')['total_crimes'].mean().reset_index()
        
        fig2 = px.bar(group_crime, x='protected_group', y='total_crimes',
                     title='Average Crimes by Protected Group',
                     color='protected_group',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig2.update_layout(
            xaxis_title='Protected Group',
            yaxis_title='Average Total Crimes'
        )
        
        fig2.write_html('dashboard_group_analysis.html')
        print("Saved group analysis dashboard: dashboard_group_analysis.html")
    
    # Correlation heatmap (interactive)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    crime_related = [col for col in numeric_cols if any(keyword in col.lower() 
                                                       for keyword in ['crime', 'rape', 'assault', 'murder', 'kidnap', 'total', 'violent', 'sexual'])]
    crime_related = [col for col in crime_related if col in df.columns][:10]  # Limit to 10 for readability
    
    if len(crime_related) > 1:
        corr_df = df[crime_related].corr()
        
        fig3 = px.imshow(corr_df, 
                        title='Correlation Heatmap of Crime Features',
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        range_color=[-1,1])
        
        fig3.update_layout(height=600)
        
        fig3.write_html('dashboard_correlations.html')
        print("Saved correlations dashboard: dashboard_correlations.html")
    
    # State-wise crime analysis
    if 'state_name' in df.columns and 'total_crimes' in df.columns:
        state_crime = df.groupby('state_name')['total_crimes'].sum().reset_index()
        top_15_states = state_crime.nlargest(15, 'total_crimes')
        
        fig4 = px.bar(top_15_states, x='total_crimes', y='state_name',
                     title='Top 15 States by Total Crimes',
                     orientation='h',
                     color='total_crimes',
                     color_continuous_scale='Bluered')
        
        fig4.update_layout(height=700)
        
        fig4.write_html('dashboard_state_analysis.html')
        print("Saved state analysis dashboard: dashboard_state_analysis.html")

def main():
    """Main function to run EDA"""
    print("Starting Exploratory Data Analysis for Crime Prediction Dataset")
    
    # Load data
    df = load_data()
    
    # Basic info
    missing_df = basic_info(df)
    
    # Analyze crime trends
    yearly_trends, yearly_stats = analyze_crime_trends(df)
    
    # Analyze protected groups if available
    if 'protected_group' in df.columns:
        group_stats = analyze_protected_groups(df)
    else:
        print("\nNo protected_group column found, skipping group analysis")
    
    # Analyze correlations
    corr_matrix = analyze_correlations(df)
    
    # Analyze states and districts
    if 'state_name' in df.columns and 'district_name' in df.columns:
        state_analysis, district_analysis = analyze_states_districts(df)
    
    # Generate interactive dashboard
    generate_eda_dashboard(df)
    
    print("\nEDA Complete!")
    print("Generated files:")
    print("- eda_crime_trends.png")
    print("- eda_protected_groups.png (if applicable)")
    print("- eda_correlations.png")
    print("- eda_states_districts.png")
    print("- dashboard_crime_trends.html")
    print("- dashboard_group_analysis.html (if applicable)") 
    print("- dashboard_correlations.html")
    print("- dashboard_state_analysis.html")

if __name__ == "__main__":
    main()