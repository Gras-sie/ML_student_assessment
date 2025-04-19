import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc 
import pandas as pd
import plotly.express as px
import pickle
import os
from dash.dependencies import Input, Output, State

current_dir = os.path.dirname(os.path.abspath(__file__))


data_path = os.path.join(current_dir, '..', 'SRC', 'Student_performance_data.csv')
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "BrightPath Academy - Student Performance Predictor"

app.layout = dbc.Container([
    dbc.Row(dbc.Col(
        html.Div([
            html.H1("BrightPath Academy", className="display-4 mb-3"),
            html.H4("Student Performance Predictor", className="text-light")
        ], className="header text-center")
    )),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Student Information", className="card-title mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Age:", className="form-label"),
                            dcc.Input(id='age', type='number', placeholder="15-18", min=15, max=18, className="form-control")
                        ], width=6),
                        dbc.Col([
                            html.Label("Gender:", className="form-label"),
                            dcc.Dropdown(
                                id='gender',
                                options=[
                                    {'label': 'Male', 'value': 0},
                                    {'label': 'Female', 'value': 1}
                                ],
                                placeholder="Select gender",
                                className="form-control"
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Ethnicity:", className="form-label"),
                            dcc.Dropdown(
                                id='ethnicity',
                                options=[
                                    {'label': 'Caucasian', 'value': 0},
                                    {'label': 'African American', 'value': 1},
                                    {'label': 'Asian', 'value': 2},
                                    {'label': 'Other', 'value': 3}
                                ],
                                placeholder="Select ethnicity",
                                className="form-control"
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Parental Education:", className="form-label"),
                            dcc.Dropdown(
                                id='parental_education',
                                options=[
                                    {'label': 'None', 'value': 0},
                                    {'label': 'High School', 'value': 1},
                                    {'label': 'Some College', 'value': 2},
                                    {'label': "Bachelor's", 'value': 3},
                                    {'label': 'Higher Study', 'value': 4}
                                ],
                                placeholder="Select education level",
                                className="form-control"
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Study Time Weekly (hours):", className="form-label"),
                            dcc.Input(id='study_time', type='number', placeholder="0-20", min=0, max=20, className="form-control")
                        ], width=6),
                        dbc.Col([
                            html.Label("Absences:", className="form-label"),
                            dcc.Input(id='absences', type='number', placeholder="0-30", min=0, max=30, className="form-control")
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Tutoring:", className="form-label"),
                            dcc.Dropdown(
                                id='tutoring',
                                options=[
                                    {'label': 'No', 'value': 0},
                                    {'label': 'Yes', 'value': 1}
                                ],
                                placeholder="Select tutoring status",
                                className="form-control"
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Parental Support:", className="form-label"),
                            dcc.Dropdown(
                                id='parental_support',
                                options=[
                                    {'label': 'None', 'value': 0},
                                    {'label': 'Low', 'value': 1},
                                    {'label': 'Medium', 'value': 2},
                                    {'label': 'High', 'value': 3},
                                    {'label': 'Very High', 'value': 4}
                                ],
                                placeholder="Select support level",
                                className="form-control"
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Extracurricular:", className="form-label"),
                            dcc.Dropdown(
                                id='extracurricular',
                                options=[
                                    {'label': 'No', 'value': 0},
                                    {'label': 'Yes', 'value': 1}
                                ],
                                placeholder="Select status",
                                className="form-control"
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Sports:", className="form-label"),
                            dcc.Dropdown(
                                id='sports',
                                options=[
                                    {'label': 'No', 'value': 0},
                                    {'label': 'Yes', 'value': 1}
                                ],
                                placeholder="Select status",
                                className="form-control"
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Music:", className="form-label"),
                            dcc.Dropdown(
                                id='music',
                                options=[
                                    {'label': 'No', 'value': 0},
                                    {'label': 'Yes', 'value': 1}
                                ],
                                placeholder="Select status",
                                className="form-control"
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Volunteering:", className="form-label"),
                            dcc.Dropdown(
                                id='volunteering',
                                options=[
                                    {'label': 'No', 'value': 0},
                                    {'label': 'Yes', 'value': 1}
                                ],
                                placeholder="Select status",
                                className="form-control"
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Button("Predict Grade", id='predict-button', color="primary", className="predict-button w-100 mt-4"),
                ])
            ], className="input-card")
        ], width=12, lg=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Prediction Result", className="card-title mb-4"),
                    dcc.Loading(
                        id="loading-prediction",
                        type="circle",
                        children=html.Div([
                            html.Div(id='prediction-output'),
                            html.Div(id='prediction-explanation', className="mt-4")
                        ])
                    )
                ])
            ], className="prediction-card"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4("Feature Analysis", className="card-title mb-4"),
                    dcc.Loading(
                        id="loading-graph",
                        type="circle",
                        children=dcc.Graph(id='feature-graph')
                    )
                ])
            ], className="feature-card mt-4")
        ], width=12, lg=6)
    ])
], fluid=True, className="app-container")

@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-explanation', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('age', 'value'),
     State('gender', 'value'),
     State('ethnicity', 'value'),
     State('parental_education', 'value'),
     State('study_time', 'value'),
     State('absences', 'value'),
     State('tutoring', 'value'),
     State('parental_support', 'value'),
     State('extracurricular', 'value'),
     State('sports', 'value'),
     State('music', 'value'),
     State('volunteering', 'value')]
)
def predict_grade(n_clicks, age, gender, ethnicity, parental_education, 
                 study_time, absences, tutoring, parental_support,
                 extracurricular, sports, music, volunteering):
    if not n_clicks or None in [age, gender, ethnicity, parental_education, 
                               study_time, absences, tutoring, parental_support,
                               extracurricular, sports, music, volunteering]:
        return html.Div("Please fill in all fields", className="text-danger"), ""
    
    try:
        model_path = os.path.join(current_dir, '..', 'Artifacts', 'PLK', 'logistic_model.pkl')
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        features = [[age, gender, ethnicity, parental_education, 
                    study_time, absences, tutoring, parental_support,
                    extracurricular, sports, music, volunteering]]
        prediction = model.predict(features)[0]
        grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
        grade = grade_map[prediction]
        grade_class = f"grade-{grade.lower()}"
        
        return [
            html.H2(f"Predicted Grade: {grade}", className=grade_class),
            html.P(get_grade_explanation(grade), className="mt-3")
        ]
    except Exception as e:
        return f"Error: {str(e)}", ""

def get_grade_explanation(grade):
    explanations = {
        'A': "Excellent performance! The student is showing strong academic capabilities.",
        'B': "Good performance. The student is performing above average.",
        'C': "Average performance. There's room for improvement in some areas.",
        'D': "Below average performance. The student may need additional support.",
        'F': "Student is struggling and needs immediate intervention and support."
    }
    return explanations.get(grade, "")

@app.callback(
    Output('feature-graph', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_graph(n_clicks):
    features = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 
                'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport',
                'Extracurricular', 'Sports', 'Music', 'Volunteering']
    correlation = data[features].corr()['StudyTimeWeekly'].sort_values()
    fig = px.bar(correlation, x=correlation.index, y=correlation.values,
                 labels={'x': 'Features', 'y': 'Correlation with Study Time'},
                 title="Feature Correlations")
    return fig

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, port=port, host="0.0.0.0")