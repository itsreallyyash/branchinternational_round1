import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("GPS Location History", style={'textAlign': 'center'}),
    
    # Control panel
    html.Div([
        # First row of controls
        html.Div([
            # Date range selector
            html.Div([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=pd.Timestamp('2017-01-01'),
                    max_date_allowed=pd.Timestamp('2018-12-31'),
                    start_date=pd.Timestamp('2017-01-01'),
                    end_date=pd.Timestamp('2017-12-31')
                )
            ], style={'flex': '1', 'marginRight': '20px'}),
            
            # Loan outcome filter
            html.Div([
                html.Label("Loan Outcome:"),
                dcc.Dropdown(
                    id='loan-outcome-filter',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Defaulted', 'value': 'defaulted'},
                        {'label': 'Repaid', 'value': 'repaid'}
                    ],
                    value='all',
                    style={'width': '200px'}
                )
            ], style={'flex': '1'}),
            
            # Accuracy threshold slider
            html.Div([
                html.Label("Max GPS Accuracy (meters):"),
                dcc.Slider(
                    id='accuracy-threshold',
                    min=0,
                    max=4000,
                    value=4000,
                    marks={0: '0m', 1000: '1000m', 2000: '2000m', 3000: '3000m', 100000000000000: '10000000000m'},
                    step=100
                )
            ], style={'flex': '2', 'marginLeft': '20px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
        
        # Statistics panel
        html.Div([
            html.Div(id='stats-panel', style={'marginTop': '10px'})
        ])
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
    # Map
    dcc.Graph(
        id='map',
        style={'height': '70vh'}
    ),
    
    # Legend
    html.Div([
        html.Div("🔴 Defaulted Loans", style={'marginRight': '20px', 'display': 'inline-block'}),
        html.Div("🔵 Repaid Loans", style={'display': 'inline-block'}),
        html.Div("Circle size indicates GPS accuracy (smaller circles = higher accuracy)", 
                 style={'marginLeft': '40px', 'display': 'inline-block'})
    ], style={'margin': '20px', 'textAlign': 'center'})
])

@app.callback(
    [Output('map', 'figure'),
     Output('stats-panel', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('loan-outcome-filter', 'value'),
     Input('accuracy-threshold', 'value')]
)
def update_map(start_date, end_date, loan_outcome, accuracy_threshold):
    # Read your CSV file
    df = pd.read_csv('processed_data.csv')  # Replace with your actual data loading
    df['gps_fix_at'] = pd.to_datetime(df['gps_fix_at'])
    
    
    # Calculate marker sizes
    max_accuracy = df['accuracy'].max()
    df['marker_size'] = 2 # Will range from 3 to 8
    
    # Create color mapping
    df['color'] = df['loan_outcome'].map({
        'defaulted': 'red',
        'repaid': 'blue'
    })
    
    # Create the map
    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        color='loan_outcome',
        color_discrete_map={
            'defaulted': 'red',
            'repaid': 'blue'
        },
        hover_data={
            'gps_fix_at': True,
            'accuracy': ':.1f',
            'loan_outcome': True,
            'user_id': True
        },
        size='marker_size',
        size_max=8,
        opacity=0.6,
        zoom=10,
        center=dict(
            lat=df['latitude'].mean(),
            lon=df['longitude'].mean()
        ),
        mapbox_style="carto-positron"
    )
    
    # Update layout
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox=dict(
            style="carto-positron",
            zoom=10
        ),
        showlegend=False  # Hide legend since we have custom legend below
    )
    
    # Calculate statistics
    stats = html.Div([
        html.H4("Current View Statistics"),
        html.Div([
            html.Div([
                f"Total Points: {len(df):,}",
                html.Br(),
                f"Unique Users: {df['user_id'].nunique():,}",
            ], style={'marginRight': '40px', 'display': 'inline-block'}),
            html.Div([
                f"Defaulted: {len(df[df['loan_outcome'] == 'defaulted']):,}",
                html.Br(),
                f"Repaid: {len(df[df['loan_outcome'] == 'repaid']):,}",
            ], style={'marginRight': '40px', 'display': 'inline-block'}),
            html.Div([
                f"Avg Accuracy: {df['accuracy'].mean():.1f}m",
                html.Br(),
                f"Time Range: {df['gps_fix_at'].min().date()} to {df['gps_fix_at'].max().date()}"
            ], style={'display': 'inline-block'})
        ])
    ])
    
    return fig, stats  # Wrap them as a tuple



if __name__ == '__main__':
    app.run_server(debug=True)