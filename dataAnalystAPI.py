from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, AsyncGenerator, Union, Callable, Tuple
import json
from datetime import datetime
import io
from pydantic import BaseModel, validator
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
from fastapi.responses import StreamingResponse
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from dataclasses import dataclass
from enum import Enum
import ast
from io import BytesIO
import base64
from concurrent.futures import ProcessPoolExecutor
import psutil
from fastapi.openapi.utils import get_openapi
import scipy
import statsmodels
import sklearn
import kaleido

# Load environment variables
env_path = Path(os.getcwd()) / '.env'
load_dotenv(env_path)

# Get environment variables
deployment_id = os.getenv('DEPLOYMENT_ID')
openai_api_key = os.getenv('DATAROBOT_API_KEY')
openai_base_url = os.getenv('OPENAI_BASE_URL')

client = OpenAI(api_key=openai_api_key, base_url=openai_base_url+deployment_id)
# client = OpenAI()

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst API",
    description="""
    An intelligent API for data analysis that provides capabilities including:
    - Data cleansing and standardization
    - Data dictionary generation
    - Question suggestions
    - Python code generation
    - Chart creation
    - Business analysis
    
    The API uses OpenAI's GPT models for intelligent analysis and response generation.
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@example.com"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    }
)

MODEL_MODE = "openai" # "openai", "gemini", anthropic

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

SYSTEM_PROMPT_GET_DICTIONARY = """
YOUR ROLE:
You are a data dictionary maker.
Inspect this metadata to decipher what each column in the dataset is about is about.
Write a short description for each column that will help an analyst effectively leverage this data in their analysis.

CONTEXT:
You will receive the following:
1) The first 10 rows of a dataframe
2) A summary of the data computed using pandas .describe()
3) For categorical data, a list of the unique values limited to the top 10 most frequent values.

CONSIDERATIONS:
The description should communicate what any acronyms might mean, what the business value of the data is, and what the analytic value might be.
You must describe ALL of the columns in the dataset to the best of your ability.

RESPONSE:
Respond with a JSON object containing the following fields:
1) columns: A list of all of the columns in the dataset
2) descriptions: A list of descriptions for each column.

"""
DICTIONARY_BATCH_SIZE = 5
SYSTEM_PROMPT_SUGGEST_A_QUESTION = """
YOUR ROLE:
Your job is to examine some meta data and suggest 3 business analytics questions that might yeild interesting insight from the data.
Inspect the user's metadata and suggest 3 different questions. They might be related, or completely unrelated to one another.
Your suggested questions might require analysis across multiple tables, or might be confined to 1 table.
Another analyst will turn your question into a SQL query. As such, your suggested question should not require advanced statistics or machine learning to answer and should be straightforward to implement in SQL.

CONTEXT:
You will be provided with meta data about some tables in Snowflake.
For each question, consider all of the tables.

YOUR RESPONSE:
Each question should be 1 or 2 sentences, no more.
Format your response as a JSON object with the following fields:
1) question1: A business question that might be answered by the data.
2) question2: A second, totally different business question that might be answered by the data.
3) question3: A third business question that touches on a different aspect of the data.

NECESSARY CONSIDERATIONS:
Do not refer to specific column names or tables in the data. Just use common language when suggesting a question. Let the next analyst figure out which columns and tables they'll need to use.
"""
SYSTEM_PROMPT_CHAT = """
ROLE:
Your job is to review a chat history between an AI assistant and a user, and possibly rephrase the user's most recent message so that it captures their complete thought in a single message. 
We will then send this message to an analytics engine for processing. 

There are a few rules to follow:
If this is the first message from the user, you should just echo it.
If this is not the first message from the user, you should decide if this most recent message constitutes a completely new question, or a revision/complication/addition to the previous question.  
If it is a revision of a previous question, you should rephrase the most recent message so that it incorporates the context of the previous question.
If it is a completely new or independent question, you should echo it.


Let me give you an example:

user: How many patients are there by race and gender?
assistant: <responds with a pandas dataframe showing the number of patients by race and gender>
user: Now sort that by patient count in ascending order
Your response: How many patients are there by race and gender sorted by patient count in ascending order?

We will then send this more complete thought to the analytics engine for processing so that we get the number of patients by race and gender sorted by patient count in ascending order

The message chain could include several requests, revisions or complications by the user. They might ask about charts, different aggregations, changes to the data, etc.
Your job is to carefully review the chain of the conversation and paraphrase the user's request so that it captures the full context of the analysis that they would like to perform.

IF THIS IS THE USER'S FIRST MESSAGE:
If this is the first/only user input message then there is no need to make any adjustments unless there is some kind of significant logical error.
In most cases, if this is the first/only message from the user, you will simply echo the user's message.
You might consider rephrasing the question so that data anlysts downstream can better understand it, but you typically should not be making any change to it.

EXAMPLE - first message:
user: Show me the sales by store, aggregated by year.
Your response: Show me the sales by store, aggregated by year.

EXAMPLE - revision of a previous question:
user: Show me the sales by store, aggregated by year.
assistant: <lists all stores, aggregated by year, with a bar chart and a line chart>
user: Instead of the bar chart, show me a pie chart
Your response: Show me the sales by store, aggregated by year. Show me a pie chart and a line chart.

EXAMPLE - completely new question:
user: Show me the sales by store, aggregated by year.
assistant: <lists all stores, aggregated by year, with a bar chart and a line chart>
user: Show me the sales by store, aggregated by year. Show me a pie chart and a line chart.
assistant: <lists all stores, aggregated by year, with a pie chart and a line chart>
user: Perform an analysis of the P&L by store
Your response: Perform an analysis of the P&L by store

YOUR RESPONSE:
Respond with JSON where there are 2 fields:
1) original_user_message: the most recent message from the user, unchanged
2) enhanced_user_message: make the changes the user's message based on the guidelines provided

CONSIDERATIONS:
You may not need to make any changes to the user's most recent message if it is their only message or if it contains a complete independent request that requires no context.
You must also consider the assistant's responses to the the user's questions. 
"""
SYSTEM_PROMPT_PYTHON_ANALYST = """
ROLE:
Your job is to write a Python function that analyzes one or more input dataframes, performing the necessary merges, calculations and aggregations required to answer the user's business question.
Carefully inspect the datasets and metadata provided to ensure your code will execute against the data and return a single Pandas dataframe containing the data relevant to the user's question.
Your function should return a dataframe that not only answers the question, but provides the necessary context so the user can fully understand the answer.
For example, if the user asks, "Which State has the highest revenue?" Your function might return the top 10 states by revenue sorted in descending order.
This way the user can analyze the context of the answer. It should also return other columns that are relevant to the question, providing additional context.

CONTEXT:
The user will provide:
1. A dictionary of dataframes (dfs) where keys are dataset names and values are the dataframes
2. A dict of data dictionaries that describe the columns across all dataframes
3. A business question to answer

YOUR RESPONSE:
Your response shall only contain a Python function called analyze_data(dfs) that takes a dictionary of dataframes as input and returns the relevant data as a single dataframe.
Your response shall be formatted as JSON with the following fields:
1) code: A string of python code that will execute and return a single pandas dataframe.
2) description: A brief description of how the code works, and how the results can be interpreted to answer the question.

For example:

def analyze_data(dfs):
    import pandas as pd
    import numpy as np
    # High level explanation 
    # of what the code does
    # should be included at the top of the function
    
    # Access individual dataframes by name
    df = dfs['dataset_name']  # Access specific dataset
    
    # Perform analysis
    # Join/merge datasets if needed
    # Compute metrics and aggregations
    
    return result_df

NECESSARY CONSIDERATIONS:
- The input dfs is a dictionary of pandas DataFrames where keys are dataset names
- Access dataframes using their names as dictionary keys, e.g. dfs['dataset_name']
- Your code should handle cases where some expected columns might be in different dataframes
- Consider appropriate joins/merges between dataframes when needed
- Document the code with comments at the top of the function explaining at a high level what the code does
- Include comments at each step to explain the code in more detail
- The function must return a single DataFrame with the analysis results
- The function shall not return a list of dataframes, a dict of dataframes, or anything other than a single dataframe.
- You may perform advanced analysis using statsmodels, scipy, numpy, pandas and scikit-learn.
...
"""
SYSTEM_PROMPT_PLOTLY_CHART = """
ROLE:
You are a data visualization expert with a focus on Python and Plotly.
Your task is to create a Python function that returns 2 complementary Plotly visualizations designed to answer a business question.
Carefully review the metadata about the columns in the dataframe to help you choose the right chart type and properly construct the chart using plotly without making mistakes.
The metadata will contain information such as the names and data types of the columns in the dataset that your charts will run against. Therefor, only refer to columns that specifically noted in the metadata. 
Choose charts types that not only complement each other superficially, but provide a comprehensive view of the data and deeper insights into the data. 
Plotly has a feature called subplots that allows you to create multiple charts in a single figure which can be useful for showing metrics for different groups or categories. 
So for example, you could make 2 complementary figures by having an aggregated view of the data in the first figure, and a more detailed breakdown by category in the second figure by using subplots. Only use subplots for 4 or fewer categories.

CONTEXT:
You will be given:
1. A business question
2. A pandas DataFrame containing the data relevant to the question
3. Metadata about the columns in the dataframe to help you choose the right chart type and properly construct the chart using plotly without making mistakes. You may only reference column names that actually are listed in the metadata!

YOUR RESPONSE:
Your response must be a Python function that returns 2 plotly.graph_objects.Figure objects.
Your function will accept a pandas DataFrame as input.
Respond with JSON with the following fields:
1) code: A string of python code that will execute and return 2 Plotly visualizations.
2) description: A brief description of how the code works, and how the results can be interpreted to answer the question.

FUNCTION REQUIREMENTS:
Name: create_charts()
Input: A pandas DataFrame containing the data relevant to the question
Output: Two plotly.graph_objects.Figure objects
Import required libraries within the function.

EXAMPLE CODE STRUCTURE:
def create_charts(df):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
     
    # Your visualization code here
    # Create two complementary visualizations
    
    return fig1, fig2

NECESSARY CONSIDERATIONS:
The input df is a pandas DataFrame that is described by the included metadata
Choose visualizations that effectively display the data and complement each other
ONLY REFER TO COLUMNS THAT ACTUALLY EXIST IN THE METADATA.
When using subplots, only use subplots for 4 or fewer categories.
You must never refer to columns that will not exist in the input dataframe.
When referring to columns in your code, spell them EXACTLY as they appear in the pandas dataframe according to the provided metadata - this might be different from how they are referenced in the business question! 
For example, if the question asks "What is the total amount paid ("AMTPAID") for each type of order?" but the metadata does not contain "AMTPAID" but rather "TOTAL_AMTPAID", you should use "TOTAL_AMTPAID" in your code because that's the column name in the data.
Data Availability: If some data is missing, plot what you can in the most sensible way.
Package Imports: If your code requires a package to run, such as statsmodels, numpy, scipy, etc, you must import the package within your function.

Data Handling:
If there are more than 100 rows, consider grouping or aggregating data for clarity.
Round values to 2 decimal places if they have more than 2.

Visualization Principles:
Choose visualizations that effectively display the data and complement each other.

Examples:
Gauge Chart and Choropleth: Display a key metric (e.g., national unemployment rate) using a gauge chart and show its variation across regions with a choropleth (e.g., state-level unemployment).
Scatter Plot and Contour Plot: Combine scatter plots for individual data points with contour plots to visualize density gradients or clustering trends (e.g., customer locations vs. density).
Bar Chart and Line Chart: Use a bar chart for categorical comparisons (e.g., monthly revenue) and overlay a line chart to illustrate trends or cumulative growth.
Choropleth and Treemap: Use a choropleth to show regional data (e.g., population by state) and a treemap to display hierarchical contributions (e.g., city-level population).
OpenStreetMap and Bubble Chart: Overlay a bubble chart on OpenStreetMap to represent multi-dimensional data points (e.g., branch size and revenue growth by location).
Pie Chart and Sunburst Chart: Show high-level proportions with a pie chart (e.g., sales by region) and dive deeper into hierarchical relationships using a sunburst chart (e.g., product-level breakdown within each region).
Scatter Plot and Histogram: Combine scatter plots to show relationships between variables with histograms to analyze frequency distributions (e.g., income vs. education level and distribution of income ranges).
Bubble Chart and Sankey Diagram: Use a bubble chart for multi-dimensional comparisons (e.g., customer spending vs. loyalty scores) and a Sankey diagram to visualize flow relationships (e.g., customer journey stages).
Choropleth and Indicator Chart: Highlight overall metrics with an indicator chart (e.g., average national GDP) and show spatial variations with a choropleth (e.g., GDP by state).
Line Chart and Area Chart: Pair a line chart to show temporal trends (e.g., sales over months) with an area chart to emphasize cumulative totals or overlapping data.
Treemap and Parallel Coordinates Plot: Use a treemap for hierarchical data visualization (e.g., sales by category and subcategory) and a parallel coordinates plot to analyze relationships between multiple attributes (e.g., sales, profit margin, and costs).
Scatter Geo and Choropleth: Use scatter geo plots to mark specific data points (e.g., retail store locations) and a choropleth to highlight regional metrics (e.g., revenue per capita).Design Guidelines:
Avoid Box and Whisker plots unless it's highly appropriate for the data or the user specifically requests it.
Avoid heatmaps unless it's highly appropriate for the data or the user specifically requests it.

Simple, not overly busy or complex.
No background colors or themes; use the default theme.

Use DataRobot Brand Colors
Primary Colors:
DataRobot Green:
HEX: #81FBA5
DataRobot Blue:
HEX: #44BFFC
DataRobot Yellow (use very sparingly, if at all):
HEX: #FFFF54
DataRobot Purple:
HEX: #909BF5
Accent Colors:
Green Variants:
Light Green: HEX #BFFD7E
Dark Green: HEX #86DAC0, #8AC2D5
Blue Variants:
Light Blue: HEX #4CCCEA
Teal: HEX #61D7CF
Yellow Variant:
Lime Yellow: HEX #EDFE60
Purple Variants:
Light Purple: HEX #8080F0, #746AFC
Deep Purple: HEX #5C41FF
Neutral Colors:
White:
HEX: #FFFFFF
Black:
HEX: #0B0B0B
Grey Variants:
Light Grey: HEX #E4E4E4, #A2A2A2
Dark Grey: HEX #6C6A6B, #231F20
Suggested Usage in Charts
Based on the color pairings and branding guidelines, here are my suggestions for using these colors in charts:

Primary Colors for Data Differentiation:

Use DataRobot Green (#81FBA5) and DataRobot Blue (#44BFFC) for major categories or distinct data series.
Use DataRobot Yellow (#FFFF54) for highlighting or calling attention to key points, but avoid using yellow
DataRobot Purple (#909BF5) can be used to differentiate less critical data or secondary information.
Accent Colors for Detailed Insights:

Variants like Light Green and Teal can be used to represent related data that needs to be distinguished from the primary green or blue.
Purple Variants (Light Purple or Deep Purple) can be used to show comparison data alongside primary categories without overwhelming the viewer.
Yellow Variants can also serve as an accent to highlight notable metrics or trends in the data, but should mostly be avoided.
Neutral Colors for Background and Context:

Black (#0B0B0B) can be used for text labels, axis lines, and borders to maintain readability.
Grey Variants like Light Grey (#E4E4E4) can be used for gridlines or background elements to add structure without distracting from the data.
Color Pairings for Emphasis:

Use the pairing combinations as shown (Green/Black/Grey, Purple/Black/Grey, etc.) to maintain consistency with brand visual identity. These pairings can be applied to legends, titles, and annotations in charts to enhance readability while sticking to the brand.

Robustness:
Ensure the function is free of syntax errors and logical problems.
Handle errors gracefully and ensure type casting for data integrity.

REATTEMPT:
If your chart code fails to execute, you will also be provided with the failed code and the error message.
Take error message into consideration when reattempting your chart code so that the problem doesn't happen again.
Try again, but don't fail this time.
"""
SYSTEM_PROMPT_BUSINESS_ANALYSIS = """
ROLE:
You are a business analyst.
Your job is to write an answer to the user's question in 3 sections: The Bottom Line, Additional Insights, Follow Up Questions.

The Bottom Line
Based on the context information provided, clearly and succinctly answer the user's question in plain language, tailored for 
someone with a business background rather than a technical one.

Additional Insights
This section is all about the "why". Discuss the underlying reasons or causes for the answer in "The Bottom Line" section. This section, 
while still business focused, should go a level deeper to help the user understand a possible root cause. Where possible, justify your answer 
using data or information from the dataset. 
Provide a bullet list, or numbered list of insights, reasons, root causes or justifications for your answer. 
Provide business advice based on the outcome noted in "The Bottom Line" section.
Suggest specific additional analyses based on the context of the question and the data available in the provided dataset.
Offer actionable recommendations. 
For example, if the data shows a declining trend in TOTAL_PROFIT, advise on potential areas to 
investigate using other data in the dataset, and propose analytics strategies to gain insights that might improve profitability.
Use markdown to format your repsonse for readability. While you might organize this content into sections, don't use headings with large

Follow Up Questions
Offer 2 or 3 follow up questions the user could ask to get deeper insight into the issue in another round of question and answer.
When you word these questions, do not use pronouns to refer to the data - always use specific column names. Only refer to data that 
that is described in the data dictionary. For example, don't refer to "sales volume" if there is no "sales volume" column.

CONTEXT:
The user has provided a business question and a dataset containing information relevant to the question.
You will also be provided with a data dictionary that describes the underlying data from which this dataset was derived. 
Based solely on the content within the provided data dictionary, you may suggest analysing other data that might be relevant or helpful for shedding more light on the topic raised by the user.
Do not suggest analysing data outside of the scope of this data dictionary.

YOUR RESPONSE:
Your response should be output as a JSON object with the following fields:
1) bottom_line: A concise answer to the user's question in plain language, tailored for someone with a business background rather than a technical one. Formatted in markdown.
2) additional_insights: A discussion of the underlying reasons or causes for the answer in "The Bottom Line" section. This section, while still business focused, should go a level deeper to help the user understand a possible root cause. Formatted in markdown.
3) follow_up_questions: A list of 3 helpful follow up questions that would lead to deeper insight into the issue in another round of analysis. When you word these questions, do not use pronouns to refer to the data - always use specific column names. Only refer to data that actually exists in the provided dataset. For example, don't refer to "sales volume" if there is no "sales volume" column.


"""

# Add custom exceptions at the top of the file
class NumericCleaningError(Exception):
    """Raised when numeric cleaning fails"""
    pass

class DateCleaningError(Exception):
    """Raised when date parsing fails"""
    pass

class CategoryCleaningError(Exception):
    """Raised when categorical cleaning fails"""
    pass

class EmptyDataError(Exception):
    """Raised when cleaning results in empty data"""
    pass

class DatasetInput(BaseModel):
    name: str
    data: List[Dict[str, Any]]
    
class CleanseRequest(BaseModel):
    datasets: List[DatasetInput]
    
class CleansingReport(BaseModel):
    columns_cleaned: List[str]
    value_counts: Dict[str, int]
    errors: List[str]
    warnings: List[str]
    
class DatasetOutput(BaseModel):
    name: str
    data: List[Dict[str, Any]]
    cleaning_report: CleansingReport
    
class CleanseResponse(BaseModel):
    datasets: List[DatasetOutput]
    metadata: Dict[str, Any]

class DictionaryRequest(BaseModel):
    data: List[Dict[str, Any]]
    
    @validator('data')
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        return v

# Add after the DictionaryRequest class
class DictionaryResponse(BaseModel):
    """Validates LLM responses for data dictionary generation
    
    Attributes:
        columns: List of column names
        descriptions: List of column descriptions
        
    Raises:
        ValueError: If validation fails
    """
    columns: List[str]
    descriptions: List[str]
    
    @validator('descriptions')
    def validate_descriptions(cls, v, values):
        # Check if columns exists in values
        if 'columns' not in values:
            raise ValueError("Columns must be provided before descriptions")
            
        # Check if lengths match
        if len(v) != len(values['columns']):
            raise ValueError(
                f"Number of descriptions ({len(v)}) must match number of columns ({len(values['columns'])})"
            )
            
        # Validate each description
        for desc in v:
            if not desc or not isinstance(desc, str):
                raise ValueError("Each description must be a non-empty string")
            if len(desc.strip()) < 10:
                raise ValueError("Descriptions must be at least 10 characters long")
                
        return v
    
    @validator('columns')
    def validate_columns(cls, v):
        if not v:
            raise ValueError("Columns list cannot be empty")
            
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate column names are not allowed")
            
        # Validate each column name
        for col in v:
            if not col or not isinstance(col, str):
                raise ValueError("Each column name must be a non-empty string")
                
        return v
    
    def to_dict(self) -> Dict[str, str]:
        """Convert columns and descriptions to dictionary format
        
        Returns:
            Dict mapping column names to their descriptions
        """
        return dict(zip(self.columns, self.descriptions))

def convert_to_datetime(value: Any, column: str) -> Optional[datetime]:
    """Convert a value to datetime with flexible format handling
    
    Args:
        value: Value to convert
        column: Column name for error reporting
        
    Returns:
        datetime or None if conversion fails
    """
    if pd.isna(value):
        return None
        
    try:
        # First try pandas to_datetime with coerce
        result = pd.to_datetime(value, infer_datetime_format=True)
        # Convert Timestamp to datetime
        if isinstance(result, pd.Timestamp):
            return result.to_pydatetime()
        return result
    except:
        try:
            # Try dateutil parser as fallback
            from dateutil import parser
            parsed = parser.parse(str(value))
            # Ensure we return a datetime object
            return parsed.replace(tzinfo=None)
        except:
            return None

@app.post("/cleanse_dataframes",
    response_model=CleanseResponse,
    summary="Cleanse and standardize multiple datasets",
    description="""
    Clean and standardize multiple pandas DataFrames with progress reporting.
    
    The endpoint handles:
    - Column name standardization
    - Numeric data cleaning
    - Date format standardization
    - Categorical data cleaning
    
    Returns a detailed cleaning report for each dataset.
    """,
    response_description="Cleaned datasets with cleaning reports",
    tags=["Data Cleaning"]
)
async def cleanse_dataframes(
    request: CleanseRequest,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> CleanseResponse:
    """
    Clean and standardize multiple pandas DataFrames.
    
    Parameters:
    - request: CleanseRequest containing datasets to clean
    - progress_callback: Optional callback for progress reporting
    
    Returns:
    - CleanseResponse containing cleaned datasets and metadata
    
    Raises:
    - HTTPException: If cleaning fails
    """
    try:
        logging.info("Starting cleanse_dataframes")
        cleaned_datasets = []
        total_datasets = len(request.datasets)
        
        for idx, dataset in enumerate(request.datasets):
            try:
                logging.info(f"Processing dataset: {dataset.name}")
                
                # Convert JSON to DataFrame
                df = pd.DataFrame(dataset.data)
                logging.debug(f"Created DataFrame with shape: {df.shape}")
                
                if df.empty:
                    raise EmptyDataError("Input DataFrame is empty")

                # Initialize cleaning report
                cleaning_report = CleansingReport(
                    columns_cleaned=[],
                    value_counts={},
                    errors=[],
                    warnings=[]
                )

                # Clean column names - only remove leading/trailing whitespace and consecutive spaces
                original_columns = df.columns.tolist()
                df.columns = [re.sub(r'\s+', ' ', col.strip()) for col in df.columns]
                cleaned_columns = df.columns.tolist()

                # Track column name changes
                for orig, cleaned in zip(original_columns, cleaned_columns):
                    if orig != cleaned:
                        cleaning_report.columns_cleaned.append(orig)
                        cleaning_report.warnings.append(
                            f"Column '{orig}' renamed to '{cleaned}'"
                        )

                # Process each column
                for column in df.columns:
                    try:
                        # Store original value counts for reporting
                        original_counts = df[column].value_counts().to_dict()
                        
                        # Clean numeric columns - more careful detection
                        if pd.api.types.is_numeric_dtype(df[column]):
                            try:
                                # Handle already numeric columns
                                df[column] = pd.to_numeric(df[column], errors='coerce')
                                
                            except Exception as e:
                                cleaning_report.errors.append(f"Error cleaning numeric column {column}: {str(e)}")
                                continue
                        # Handle columns that might be numeric strings with currency/percentage
                        elif (df[column].dtype == 'object' and 
                              df[column].notna().all() and  # Only check non-null values
                              df[column].str.replace(r'[$%,\s]', '', regex=True).str.match(r'^-?\d*\.?\d*$').all()):
                            try:
                                # Remove currency symbols, commas, and percentages
                                df[column] = pd.to_numeric(
                                    df[column].astype(str).str.replace(r'[$%,\s]', '', regex=True),
                                    errors='coerce'
                                )
                                
                            except Exception as e:
                                cleaning_report.errors.append(f"Error cleaning numeric column {column}: {str(e)}")
                                continue

                        # Clean date columns
                        elif is_date_column(df[column]):
                            try:
                                original_values = df[column].copy()
                                # Convert to datetime strings using vectorized operation
                                df[column] = convert_datetime_series(df[column])
                                
                                # Compare before and after
                                if not df[column].equals(original_values):
                                    cleaning_report.columns_cleaned.append(column)
                                    cleaning_report.value_counts[column] = {
                                        'before': {
                                            str(k): str(v) for k, v in original_counts.items()
                                        },
                                        'after': df[column].value_counts().to_dict(),  # Already strings
                                        'change_type': 'date_cleaning'
                                    }
                            except Exception as e:
                                cleaning_report.errors.append(f"Error cleaning date column {column}: {str(e)}")
                                continue

                        # Clean categorical columns
                        elif df[column].dtype == 'object':
                            try:
                                original_values = df[column].copy()
                                
                                # Handle non-null values only
                                mask = df[column].notna()
                                if mask.any():  # Only process if there are any non-null values
                                    # Convert to string only if not already string
                                    temp_series = df.loc[mask, column]
                                    if not pd.api.types.is_string_dtype(temp_series):
                                        temp_series = temp_series.astype(str)
                                    
                                    # Only strip leading/trailing spaces, preserve internal spaces
                                    df.loc[mask, column] = temp_series.str.strip()
                                
                                # Compare before and after
                                if not df[column].equals(original_values):
                                    cleaning_report.columns_cleaned.append(column)
                                    cleaning_report.value_counts[column] = {
                                        'before': original_counts,
                                        'after': df[column].value_counts().to_dict(),
                                        'change_type': 'categorical_cleaning'
                                    }
                            except Exception as e:
                                cleaning_report.errors.append(f"Error cleaning categorical column {column}: {str(e)}")
                                continue

                    except Exception as e:
                        cleaning_report.errors.append(f"Error processing column {column}: {str(e)}")
                        continue

                # Create DatasetOutput - ensure all data is JSON serializable
                cleaned_dataset = DatasetOutput(
                    name=dataset.name,
                    data=df.replace({pd.NaT: None}).to_dict('records'),  # Replace NaT with None
                    cleaning_report=cleaning_report
                )
                cleaned_datasets.append(cleaned_dataset)
                logging.info(f"Successfully cleaned dataset: {dataset.name}")

                # Report progress if callback provided
                if progress_callback:
                    progress = int((idx + 1) / total_datasets * 100)
                    await progress_callback(f"Processed {dataset.name}", progress)

            except Exception as e:
                logging.error(f"Error processing dataset {dataset.name}: {str(e)}")
                raise

        return CleanseResponse(
            datasets=cleaned_datasets,
            metadata={
                "total_datasets": total_datasets,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
        )

    except Exception as e:
        logging.error(f"Error in cleanse_dataframes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache key generator for DataFrames
def generate_df_hash(df: pd.DataFrame) -> str:
    """Generate a hash key for DataFrame caching based on content"""
    # Get sample of data and column info for hash
    sample = df.head(100).to_json()
    cols = ','.join(df.columns)
    dtypes = ','.join(df.dtypes.astype(str))
    
    # Create hash
    hash_input = f"{sample}{cols}{dtypes}".encode()
    return hashlib.md5(hash_input).hexdigest()

def process_column_batch(
    columns: List[str], 
    df: pd.DataFrame,
    batch_size: int = 5
) -> Dict[str, str]:
    """Process a batch of columns to get their descriptions"""
    
    # Get sample data and stats for just these columns
    # Convert timestamps to ISO format strings for JSON serialization
    sample_data = {}
    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # Convert timestamps to ISO format strings
            sample_data[col] = df[col].head(10).apply(
                lambda x: x.isoformat() if pd.notnull(x) else None
            ).to_dict()
        else:
            sample_data[col] = df[col].head(10).to_dict()

    # Handle numeric summary
    numeric_summary = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            numeric_summary[col] = {
                k: float(v) if pd.notnull(v) else None 
                for k, v in desc.to_dict().items()
            }
    
    # Get categories for non-numeric columns
    categories = []
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            try:
                value_counts = df[column].value_counts().head(10)
                # Convert any timestamp values to strings
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    value_counts.index = value_counts.index.map(
                        lambda x: x.isoformat() if pd.notnull(x) else None
                    )
                categories.append({column: list(value_counts.keys())})
            except:
                continue

    # Create messages for OpenAI
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GET_DICTIONARY},
        {"role": "user", "content": f"Data: {json.dumps(sample_data)}"},
        {"role": "user", "content": f"Statistical Summary: {json.dumps(numeric_summary)}"}
    ]
    
    if categories:
        messages.append({"role": "user", "content": f"Categorical Values: {json.dumps(categories)}"})

    # Get descriptions from OpenAI
    if MODEL_MODE == "openai":
        completion = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages,
            response_format={"type": "json_object"},         
            stream=False
        )    
        response = json.loads(completion.choices[0].message.content)
    elif MODEL_MODE in ["gemini", "anthropic"]:
        completion = client.chat.completions.create(
            model="gemini-1.5-pro", # or appropriate model name
            messages=messages
        )
        # Extract JSON from response by looking for ```json blocks
        content = completion.choices[0].message.content
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            response = json.loads(json_match.group(1))
        else:
            raise ValueError("No JSON block found in model response")
    
    try:
        # Validate response using DictionaryResponse
        validated = DictionaryResponse(
            columns=response.get('columns', []),
            descriptions=response.get('descriptions', [])
        )
        
        # Convert to dictionary format
        descriptions = validated.to_dict()
        
        # Only return descriptions for requested columns
        return {col: descriptions.get(col, "No description available") for col in columns}
        
    except ValueError as e:
        logging.error(f"Invalid dictionary response: {str(e)}")
        # Fallback: return basic descriptions
        return {col: "No valid description available" for col in columns}

@app.post("/get_dictionary",
    response_model=Dict[str, Any],
    summary="Generate data dictionary",
    description="""
    Generate comprehensive data dictionary for multiple datasets.
    
    The endpoint:
    - Analyzes column metadata
    - Generates column descriptions
    - Provides data types and sample values
    - Handles parallel processing for large datasets
    
    Returns detailed dictionary entries for all columns.
    """,
    response_description="Data dictionary with column descriptions",
    tags=["Data Dictionary"]
)
async def get_dictionary(request: DictionaryRequest) -> Dict[str, Any]:
    """
    Generate data dictionary for multiple datasets.
    
    Parameters:
    - request: DictionaryRequest containing datasets
    
    Returns:
    - Dictionary containing column descriptions and metadata
    
    Raises:
    - HTTPException: If dictionary generation fails
    """
    try:
        # Add debug logging
        logging.info(f"Received dictionary request with {len(request.datasets)} datasets")
        
        metadata = {
            "total_datasets": len(request.datasets),
            "processing_start": datetime.now().isoformat(),
            "batch_times": [],
            "errors": []
        }

        # Process datasets using ThreadPoolExecutor instead of ProcessPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Map datasets to futures
            dataset_futures = {
                executor.submit(process_dataset, dataset): dataset.name 
                for dataset in request.datasets
            }
            
            # Add debug logging
            logging.info(f"Created {len(dataset_futures)} dataset futures")
            
            # Collect results as they complete
            results = []
            for future in as_completed(dataset_futures):
                dataset_name = dataset_futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    metadata["batch_times"].append(result["batch_time"])
                    logging.info(f"Processed dataset {dataset_name} with {len(result.get('dictionary', []))} entries")
                except Exception as e:
                    error_msg = f"Error processing dataset {dataset_name}: {str(e)}"
                    logging.error(error_msg)
                    metadata["errors"].append(error_msg)
                    results.append({
                        "name": dataset_name,
                        "dictionary": [],
                        "cache_hit": False,
                        "error": error_msg
                    })

        metadata["processing_end"] = datetime.now().isoformat()
        metadata["total_time"] = (
            datetime.fromisoformat(metadata["processing_end"]) - 
            datetime.fromisoformat(metadata["processing_start"])
        ).total_seconds()
            
        response = {
            "dictionaries": results,
            "metadata": metadata
        }
        logging.info(f"Returning dictionary response with {len(results)} results")
        return response
            
    except Exception as e:
        logging.error(f"Error in get_dictionary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def process_dataset(dataset: DatasetInput) -> Dict[str, Any]:
    """Process a single dataset with parallel column batch processing"""
    try:
        batch_start = datetime.now()
        
        # Convert JSON to DataFrame
        df = pd.DataFrame(dataset.data)
        
        # Add debug logging
        logging.info(f"Processing dataset {dataset.name} with shape {df.shape}")
        
        # Handle empty dataset
        if df.empty:
            logging.warning(f"Dataset {dataset.name} is empty")
            return {
                "name": dataset.name,
                "dictionary": [],
                "cache_hit": False,
                "batch_time": 0
            }
        
        # Generate cache key
        df_hash = generate_df_hash(df)
        
        # Split columns into batches
        column_batches = [
            list(df.columns[i:i+DICTIONARY_BATCH_SIZE]) 
            for i in range(0, len(df.columns), DICTIONARY_BATCH_SIZE)
        ]
        logging.info(f"Created {len(column_batches)} batches for {len(df.columns)} columns")
        
        # Process column batches using ThreadPoolExecutor
        batch_results = {}  # Change to dictionary to maintain column-description mapping
        with ThreadPoolExecutor() as executor:
            batch_futures = {
                executor.submit(
                    process_column_batch, 
                    batch, 
                    df,
                    DICTIONARY_BATCH_SIZE
                ): batch 
                for batch in column_batches
            }
            
            # Collect results as they complete
            for future in as_completed(batch_futures):
                try:
                    result = future.result()
                    # Assuming process_column_batch returns a dictionary mapping columns to descriptions
                    batch_results.update(result)  # Merge results maintaining column mapping
                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    continue
        
        # Combine results
        dictionary = [
            {
                "data_type": str(df[col].dtype),
                "column": col,
                "description": batch_results.get(col, "No description available")
            }
            for col in df.columns
        ]
        
        logging.info(f"Created dictionary with {len(dictionary)} entries for dataset {dataset.name}")
        
        batch_time = (datetime.now() - batch_start).total_seconds()
        
        return {
            "name": dataset.name,
            "dictionary": dictionary,
            "cache_hit": False,
            "batch_time": batch_time
        }
        
    except Exception as e:
        logging.error(f"Error processing dataset {dataset.name}: {str(e)}")
        raise Exception(f"Error processing dataset {dataset.name}: {str(e)}")

# Add memory management helper
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss / 1024 / 1024,  # RSS in MB
        "vms": memory_info.vms / 1024 / 1024,  # VMS in MB
        "percent": process.memory_percent()
    }

@dataclass
class QuestionValidationResult:
    """Stores validation results for suggested questions"""
    question: str
    is_valid: bool
    available_columns: List[str]
    missing_columns: List[str]
    validation_message: str

def validate_question_feasibility(
    question: str, 
    available_columns: List[str]
) -> QuestionValidationResult:
    """Validate if a question can be answered with available data
    
    Checks if common data elements mentioned in the question exist in columns
    """
    # Convert question and columns to lowercase for matching
    question_lower = question.lower()
    columns_lower = [col.lower() for col in available_columns]
    
    # Extract potential column references from question
    words = set(re.findall(r'\b\w+\b', question_lower))
    
    # Find matches and missing terms
    found_columns = [col for col in columns_lower if any(word in col for word in words)]
    missing_columns = [word for word in words if any(
        word in col for col in columns_lower
    )]
    
    is_valid = len(found_columns) > 0
    message = (
        "Question can be answered with available data" 
        if is_valid 
        else "Question may require unavailable data"
    )
    
    return QuestionValidationResult(
        question=question,
        is_valid=is_valid,
        available_columns=found_columns,
        missing_columns=missing_columns,
        validation_message=message
    )

async def generate_question_suggestions(dictionary: pd.DataFrame, max_columns: int = 40) -> Dict[str, Any]:
    """Generate and validate suggested analysis questions
    
    Args:
        dictionary: DataFrame containing data dictionary
        max_columns: Maximum number of columns to include in prompt
        
    Returns:
        Dict containing:
            - questions: List of validated question objects
            - metadata: Dictionary of processing information
    """
    try:
        # Validate input
        if dictionary.empty:
            raise ValueError("Dictionary DataFrame cannot be empty")
            
        required_cols = ['column', 'description', 'data_type']
        if not all(col in dictionary.columns for col in required_cols):
            raise ValueError(f"Dictionary must contain columns: {required_cols}")
            
        # Limit columns for OpenAI prompt
        total_columns = len(dictionary)
        if total_columns > max_columns:
            # Take first and last 20 columns
            half_max = max_columns // 2
            first_half = dictionary.head(half_max)
            last_half = dictionary.tail(half_max)
            
            # Remove any duplicates
            dictionary = pd.concat([first_half, last_half]).drop_duplicates()
        
        # Convert dictionary to format expected by OpenAI
        dict_data = {
            'columns': dictionary['column'].tolist(),
            'descriptions': dictionary['description'].tolist(),
            'data_types': dictionary['data_type'].tolist()
        }
        
        # Create OpenAI messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUGGEST_A_QUESTION},
            {"role": "user", "content": f"Data Dictionary:\n{json.dumps(dict_data)}"}
        ]
        
        # Get suggestions from OpenAI        
        if MODEL_MODE == "openai":
            completion = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=messages,
                response_format={"type": "json_object"},         
                stream=False
            )    
            response = json.loads(completion.choices[0].message.content)
        elif MODEL_MODE in ["gemini", "anthropic"]:
            completion = client.chat.completions.create(
                model="gemini-1.5-pro", # or appropriate model name
                messages=messages
            )
            # Extract JSON from response by looking for ```json blocks
            content = completion.choices[0].message.content
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                response = json.loads(json_match.group(1))
            else:
                raise ValueError("No JSON block found in model response")
        
        # Validate each suggested question
        available_columns = dictionary['column'].tolist()
        validated_questions = []
        
        for key in ['question1', 'question2', 'question3']:
            if question := response.get(key):
                validation = validate_question_feasibility(
                    question, 
                    available_columns
                )
                validated_questions.append({
                    'question': validation.question,
                    'is_valid': validation.is_valid,
                    'available_columns': validation.available_columns,
                    'missing_columns': validation.missing_columns,
                    'validation_message': validation.validation_message
                })
        
        # Prepare metadata
        metadata = {
            'total_columns': total_columns,
            'columns_used': len(dictionary),
            'timestamp': datetime.now().isoformat(),
            'questions_generated': len(validated_questions),
            'valid_questions': sum(1 for q in validated_questions if q['is_valid'])
        }
        
        return {
            'questions': validated_questions,
            'metadata': metadata
        }
            
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest_questions",
    response_model=Dict[str, Any],
    summary="Suggest analysis questions",
    description="""
    Generate and validate suggested analysis questions based on available data.
    
    The endpoint:
    - Analyzes available columns
    - Suggests relevant business questions
    - Validates question feasibility
    - Provides context for each suggestion
    
    Returns validated questions with metadata.
    """,
    response_description="Suggested analysis questions with validation",
    tags=["Question Generation"]
)
async def suggest_questions(request: DictionaryRequest) -> Dict[str, Any]:
    """
    Generate and validate suggested analysis questions.
    
    Parameters:
    - request: DictionaryRequest containing dataset information
    
    Returns:
    - Dictionary containing suggested questions and metadata
    
    Raises:
    - HTTPException: If question generation fails
    """
    try:
        # Input validation
        if not request.datasets:
            raise ValueError("Dictionary cannot be empty")
            
        # Convert dictionary list to DataFrame
        dict_df = pd.DataFrame([
            {
                'column': f"{dataset.name}.{col}",
                'description': f"Column {col} from dataset {dataset.name}",
                'data_type': str(pd.DataFrame(dataset.data)[col].dtype)
            }
            for dataset in request.datasets
            for col in pd.DataFrame(dataset.data).columns
        ])
        
        return await generate_question_suggestions(dict_df)
            
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RunAnalysisRequest(BaseModel):
    """Request model for analysis endpoint
    
    Attributes:
        data: Dictionary of datasets, where each dataset is a list of dictionaries
        dictionary: Dictionary of data dictionaries, where each dictionary describes a dataset's columns
        question: Business question to analyze
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """
    data: Dict[str, List[Dict[str, Any]]]
    dictionary: Dict[str, List[Dict[str, Union[str, Dict[str, str]]]]]  # Allow dictionary values for description
    question: str
    error_message: Optional[str] = None
    failed_code: Optional[str] = None
    
    @validator('data')
    def validate_data(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Input data must be a dictionary of datasets")
        if not all(isinstance(dataset, list) for dataset in v.values()):
            raise ValueError("Each dataset must be a list of dictionaries")
        return v

    @validator('dictionary')
    def validate_dictionary(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Dictionary must be a dictionary of dataset descriptions")
        
        # Process dictionary values to ensure descriptions are strings
        processed = {}
        for dataset_name, descriptions in v.items():
            processed_descriptions = []
            for desc in descriptions:
                if not isinstance(desc, dict):
                    raise ValueError("Each description must be a dictionary")
                
                # Convert any dictionary values in description to strings
                processed_desc = desc.copy()
                if 'description' in desc and isinstance(desc['description'], dict):
                    # Join key-value pairs from the description dictionary
                    desc_str = '; '.join(f"{k}: {v}" for k, v in desc['description'].items())
                    processed_desc['description'] = desc_str
                
                processed_descriptions.append(processed_desc)
            processed[dataset_name] = processed_descriptions
            
        return processed

    class Config:
        arbitrary_types_allowed = True

class PythonAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]  # Changed from DataFrame to List of JSON objects
    dictionary: List[Dict[str, Any]]  # Changed from DataFrame to List of dictionary entries
    question: str
    error_message: Optional[str] = None
    failed_code: Optional[str] = None
    
    @validator('data')
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of JSON objects")
        if len(v) == 0:
            raise ValueError("Data cannot be empty")
        return v
        
    @validator('dictionary')
    def validate_dictionary(cls, v):
        if not isinstance(v, list):
            raise ValueError("Dictionary must be a list")
        required_keys = {'column', 'description', 'data_type'}
        if not all(required_keys.issubset(d.keys()) for d in v):
            raise ValueError(f"Dictionary entries must contain keys: {required_keys}")
        return v
        
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

@app.post("/get_python_analysis_code",
    response_model=Dict[str, str],
    summary="Generate Python analysis code",
    description="""
    Generate Python code to analyze data based on a business question.
    
    The endpoint:
    - Interprets the business question
    - Generates appropriate analysis code
    - Validates code safety
    - Provides execution context
    
    Returns validated Python code with description.
    """,
    response_description="Generated Python code with description",
    tags=["Code Generation"]
)
async def get_python_analysis_code(request: RunAnalysisRequest) -> Dict[str, str]:
    """
    Generate Python analysis code based on JSON data and question.
    
    Parameters:
    - request: RunAnalysisRequest containing data and question
    
    Returns:
    - Dictionary containing generated code and description
    
    Raises:
    - HTTPException: If code generation fails
    """
    try:
        # Convert dictionary data structure to list of columns for all datasets
        all_columns = []
        all_descriptions = []
        all_data_types = []
        
        for dataset_name, dictionary_list in request.dictionary.items():
            for entry in dictionary_list:
                if isinstance(entry, dict) and 'column' in entry:
                    all_columns.append(f"{dataset_name}.{entry['column']}")
                    all_descriptions.append(entry.get('description', ''))
                    all_data_types.append(entry.get('data_type', ''))

        # Create dictionary format for prompt
        dictionary_data = {
            'columns': all_columns,
            'descriptions': all_descriptions,
            'data_types': all_data_types
        }

        # Get sample data and shape info for all datasets
        all_samples = []
        all_shapes = []
        
        for dataset_name, dataset in request.data.items():
            df = pd.DataFrame(dataset)
            all_shapes.append(f"{dataset_name}: {df.shape[0]} rows x {df.shape[1]} columns")
            # Limit sample to 10 rows
            sample_df = df.head(10)
            all_samples.append(f"{dataset_name}:\n{sample_df.to_string()}")

        shape_info = "\n".join(all_shapes)
        sample_data = "\n\n".join(all_samples)

        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PYTHON_ANALYST},
            {"role": "user", "content": f"Business Question: {request.question}"},
            {"role": "user", "content": f"Data Shapes:\n{shape_info}"},
            {"role": "user", "content": f"Sample Data:\n{sample_data}"},
            {"role": "user", "content": f"Data Dictionary:\n{json.dumps(dictionary_data)}"}
        ]

        # Add error context if available
        if request.error_message and request.failed_code:
            messages.extend([
                {"role": "user", "content": "Previous attempt failed with error:"},
                {"role": "user", "content": request.error_message},
                {"role": "user", "content": "Failed code:"},
                {"role": "user", "content": request.failed_code},
                {"role": "user", "content": "Please generate new code that avoids this error."}
            ])

        # Get response from OpenAI
        if MODEL_MODE == "openai":
            completion = client.chat.completions.create(
                model="gpt-4o", 
                temperature=0.1,
                messages=messages,
                response_format={"type": "json_object"},         
                stream=False
            )    
            response = json.loads(completion.choices[0].message.content)
        elif MODEL_MODE in ["gemini", "anthropic"]:
            completion = client.chat.completions.create(
                model="gemini-1.5-pro", # or appropriate model name
                messages=messages
            )
            # Extract JSON from response by looking for ```json blocks
            content = completion.choices[0].message.content
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                response = json.loads(json_match.group(1))
            else:
                raise ValueError("No JSON block found in model response")
        
        return {
            "code": response.get("code", ""),
            "description": response.get("description", "")
        }
            
    except Exception as e:
        logging.error(f"Error generating analysis code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ChartRequest(BaseModel):
    """Request model for charts endpoint
    
    Attributes:
        data: List of dictionaries representing a single dataset
        question: Business question to visualize
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """
    data: List[Dict[str, Any]]
    question: str
    error_message: Optional[str] = None
    failed_code: Optional[str] = None
    
    @validator('data')
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        if not all(isinstance(record, dict) for record in v):
            raise ValueError("Each record must be a dictionary")
        return v

@dataclass
class ChartGenerationResult:
    """Container for chart generation results"""
    fig1: go.Figure
    fig2: go.Figure
    code: str
    validation: Dict[str, Any]
    metadata: Dict[str, Any]
    attempts: int
    validation_errors: List[str]
    execution_errors: List[Dict[str, Any]]
    code_history: List[Dict[str, Any]]

def validate_chart_code(code: str) -> Tuple[bool, str]:
    """Validate chart generation code for safety and correctness"""
    try:
        tree = ast.parse(code)
        imports = []
        
        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend(n.name.split('.')[0] for n in node.names)
                else:
                    imports.append(node.module.split('.')[0])
                    
        allowed_modules = {'pandas', 'numpy', 'plotly', 'scipy'}
        illegal_imports = set(imports) - allowed_modules
        if illegal_imports:
            return False, f"Illegal imports detected: {illegal_imports}"
        
        # Verify create_charts function exists
        has_create_charts = any(
            isinstance(node, ast.FunctionDef) and node.name == 'create_charts'
            for node in ast.walk(tree)
        )
        if not has_create_charts:
            return False, "Missing create_charts function"
            
        return True, "Validation passed"
            
    except SyntaxError as e:
        return False, f"Syntax error in code: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def figure_to_base64(fig: go.Figure) -> Optional[str]:
    """Convert Plotly figure to base64 encoded PNG"""
    try:
        if not isinstance(fig, go.Figure):
            raise ValueError(f"Expected plotly.graph_objects.Figure, got {type(fig)}")
        img_bytes = fig.to_image(format="png")
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        logging.error(f"Failed to convert figure to base64: {str(e)}")
        return None

async def create_charts(
    df: pd.DataFrame,
    question: str,
    metadata: Dict[str, Any],
    error_message: Optional[str] = None,
    failed_code: Optional[str] = None,
    max_attempts: int = 3
) -> ChartGenerationResult:
    """Generate and validate chart code with retry logic"""
    attempts = 0
    validation_errors = []
    execution_errors = []
    code_history = []
    
    while attempts < max_attempts:
        attempts += 1
        
        try:
            # Create messages for OpenAI
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_PLOTLY_CHART},
                {"role": "user", "content": f"Question: {question}"},
                {"role": "user", "content": f"Data Metadata:\n{json.dumps(metadata)}"},
                {"role": "user", "content": f"Data top 25 rows:\n{df.head(25).to_string()}"}
            ]

            # Add error context if available
            if error_message and failed_code:
                messages.extend([
                    {"role": "user", "content": f"Previous error: {error_message}"},
                    {"role": "user", "content": f"Failed code:\n{failed_code}"}
                ])

            # Get response based on model mode
            if MODEL_MODE == "openai":
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0,
                    messages=messages,
                    response_format={"type": "json_object"},
                    stream=False
                )
                response = json.loads(completion.choices[0].message.content)
            elif MODEL_MODE in ["gemini", "anthropic"]:
                completion = client.chat.completions.create(
                    model="gemini-1.5-pro", # or appropriate model name
                    messages=messages
                )
                # Extract JSON from response by looking for ```json blocks
                content = completion.choices[0].message.content
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    response = json.loads(json_match.group(1))
                else:
                    raise ValueError("No JSON block found in model response")
            
            code = response.get("code")
            
            # Track code history
            code_history.append({
                "attempt": attempts,
                "code": code,
                "timestamp": datetime.now().isoformat()
            })
            
            # Validate the generated code
            is_valid, validation_message = validate_chart_code(code)
            
            if not is_valid:
                validation_errors.append({
                    "attempt": attempts,
                    "error": validation_message,
                    "code": code,
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            try:
                # Create namespace for execution with single dataframe
                namespace = {
                    'pd': pd,
                    'np': np,
                    'df': df,  # Pass single dataframe instead of dictionary
                    'go': go,
                    'make_subplots': make_subplots
                }
                
                # Execute the code with stdout/stderr capture
                stdout = io.StringIO()
                stderr = io.StringIO()
                
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exec(code, namespace)
                    fig1, fig2 = namespace['create_charts'](df)  # Pass single dataframe
                
                return ChartGenerationResult(
                    fig1=fig1,
                    fig2=fig2,
                    code=code,
                    validation={"is_valid": True, "message": validation_message},
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "attempts": attempts,
                        "stdout": stdout.getvalue(),
                        "stderr": stderr.getvalue()
                    },
                    attempts=attempts,
                    validation_errors=validation_errors,
                    execution_errors=execution_errors,
                    code_history=code_history
                )
                
            except Exception as exec_error:
                execution_errors.append({
                    "attempt": attempts,
                    "error_type": type(exec_error).__name__,
                    "error_message": str(exec_error),
                    "code": code,
                    "stdout": stdout.getvalue() if 'stdout' in locals() else "",
                    "stderr": stderr.getvalue() if 'stderr' in locals() else "",
                    "timestamp": datetime.now().isoformat()
                })
                
                if attempts == max_attempts:
                    raise ValueError(f"Failed to execute charts after {max_attempts} attempts. Last error: {str(exec_error)}")
            
        except Exception as e:
            execution_errors.append({
                "attempt": attempts,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "code": code if 'code' in locals() else None,
                "timestamp": datetime.now().isoformat()
            })
            
            if attempts == max_attempts:
                raise ValueError(f"Failed to generate valid charts after {max_attempts} attempts: {str(e)}")

    raise ValueError(f"Failed to generate valid charts after {max_attempts} attempts")

class RunChartsRequest(BaseModel):
    """Request model for charts endpoint
    
    Attributes:
        data: List of dictionaries representing a single dataset
        question: Business question to visualize
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """
    data: List[Dict[Union[str, int], Any]]  # Allow both string and integer keys
    question: str
    error_message: Optional[str] = None
    failed_code: Optional[str] = None
    
    @validator('data')
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        if not all(isinstance(record, dict) for record in v):
            raise ValueError("Each record must be a dictionary")
            
        # Convert numeric keys to strings in nested dictionaries
        def convert_numeric_keys(d):
            if not isinstance(d, dict):
                return d
            return {
                str(k): convert_numeric_keys(v) if isinstance(v, dict) else v 
                for k, v in d.items()
            }
            
        # Convert all records
        converted = [convert_numeric_keys(record) for record in v]
        
        # Ensure all keys are strings after conversion
        for record in converted:
            if not all(isinstance(k, str) for k in record.keys()):
                raise ValueError("All dictionary keys must be strings after conversion")
                
        return converted

    class Config:
        arbitrary_types_allowed = True  # Allow any type in dictionary values

@app.post("/run_charts")
async def run_charts(request: RunChartsRequest) -> Dict[str, Any]:
    """
    Generate and execute chart code with validation.
    """
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame(request.data)
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")

        # Generate metadata about the dataframe
        metadata = {
            'metadata_shape': list(df.shape),
            'metadata_describe': json.loads(df.describe(include='all').to_json()),
            'metadata_dtypes': json.loads(df.dtypes.astype(str).to_json())
        }

        max_attempts = 3
        attempt = 0
        last_error = None
        last_failed_code = None

        while attempt < max_attempts:
            try:
                # Generate charts with retry logic
                result = await create_charts(
                    df=df.head(25),
                    question=request.question,
                    metadata=metadata,
                    error_message=last_error,
                    failed_code=last_failed_code
                )
                
                # Convert figures to base64
                fig1_base64 = figure_to_base64(result.fig1) if result.fig1 else None
                fig2_base64 = figure_to_base64(result.fig2) if result.fig2 else None
                
                return {
                    "fig1": result.fig1,
                    "fig2": result.fig2,
                    "fig1_base64": fig1_base64,
                    "fig2_base64": fig2_base64,
                    "code": result.code,
                    "metadata": {
                        **result.metadata,
                        "dataframe_metadata": metadata,
                        "validation": result.validation,
                        "attempts": result.attempts,
                        "validation_errors": result.validation_errors,
                        "execution_errors": result.execution_errors,
                        "code_history": result.code_history,
                        "performance": {
                            "memory_usage": get_memory_usage(),
                            "total_time": (
                                datetime.fromisoformat(result.metadata["timestamp"]) - 
                                datetime.fromisoformat(result.code_history[0]["timestamp"])
                            ).total_seconds()
                        }
                    }
                }

            except Exception as e:
                attempt += 1
                last_error = str(e)
                last_failed_code = result.code if 'result' in locals() else None
                
                if attempt >= max_attempts:
                    error_context = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "validation_errors": result.validation_errors if 'result' in locals() else [],
                        "execution_errors": result.execution_errors if 'result' in locals() else [],
                        "code_history": result.code_history if 'result' in locals() else [],
                        "attempts": attempt,
                        "timestamp": datetime.now().isoformat()
                    }
                    raise HTTPException(
                        status_code=500, 
                        detail={"error": str(e), "context": error_context}
                    )

    except ValueError as e:
        # Only catch and re-raise validation errors without retrying
        raise HTTPException(status_code=422, detail=str(e))

class BusinessAnalysisRequest(BaseModel):
    """Request model for business analysis endpoint
    
    Attributes:
        data: List of dictionaries representing a single dataset
        dictionary: List of dictionary entries describing columns
        question: Business question to analyze
    """
    data: List[Dict[Union[str, int], Any]]  # Allow both string and integer keys
    dictionary: List[Dict[Union[str, int], Any]]  # Allow both string and integer keys
    question: str
    
    @validator('data')
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of JSON objects")
        if len(v) == 0:
            raise ValueError("Data cannot be empty")
            
        # Convert numeric keys to strings in nested dictionaries
        def convert_numeric_keys(d):
            if not isinstance(d, dict):
                return d
            return {
                str(k): convert_numeric_keys(v) if isinstance(v, dict) else v 
                for k, v in d.items()
            }
            
        # Convert all records
        converted = [convert_numeric_keys(record) for record in v]
        
        # Ensure all keys are strings after conversion
        for record in converted:
            if not all(isinstance(k, str) for k in record.keys()):
                raise ValueError("All dictionary keys must be strings after conversion")
                
        return converted

    @validator('dictionary')
    def validate_dictionary(cls, v):
        if not isinstance(v, list):
            raise ValueError("Dictionary must be a list")
            
        # Convert numeric keys to strings in dictionary entries
        def convert_numeric_keys(d):
            if not isinstance(d, dict):
                return d
            return {
                str(k): convert_numeric_keys(v) if isinstance(v, dict) else v 
                for k, v in d.items()
            }
            
        # Convert all dictionary entries
        converted = [convert_numeric_keys(entry) for entry in v]
        
        # Validate required keys exist after conversion
        required_keys = {'column', 'description', 'data_type'}
        if not all(required_keys.issubset(d.keys()) for d in converted):
            raise ValueError(f"Dictionary entries must contain keys: {required_keys}")
            
        return converted
        
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    class Config:
        arbitrary_types_allowed = True  # Allow any type in dictionary values

@app.post("/get_business_analysis",
    response_model=Dict[str, Any],
    summary="Generate business analysis",
    description="""
    Generate comprehensive business analysis based on data and question.
    
    The endpoint provides:
    - Bottom line answer
    - Additional insights
    - Follow-up questions
    - Analysis context
    
    Returns structured analysis response.
    """,
    response_description="Business analysis with insights",
    tags=["Business Analysis"]
)
async def get_business_analysis(request: BusinessAnalysisRequest) -> Dict[str, Any]:
    """
    Generate business analysis based on data and question.
    
    Parameters:
    - request: BusinessAnalysisRequest containing data and question
    
    Returns:
    - Dictionary containing analysis components
    
    Raises:
    - HTTPException: If analysis generation fails
    """
    try:
        # Convert JSON data to DataFrame for analysis
        df = pd.DataFrame(request.data)
        
        # Get first 1000 rows as CSV with quoted values for context
        df_csv = df.head(750).to_csv(index=False, quoting=1)
        
        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BUSINESS_ANALYSIS},
            {"role": "user", "content": f"Business Question: {request.question}"},          
            {"role": "user", "content": f"Analyzed Data:\n{df_csv}"},
            {"role": "user", "content": f"Data Dictionary:\n{json.dumps(request.dictionary)}"}
        ]

        # Get response based on model mode
        if MODEL_MODE == "openai":
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=messages,
                response_format={"type": "json_object"},
                stream=False
            )
            response = json.loads(completion.choices[0].message.content)
        elif MODEL_MODE in ["gemini", "anthropic"]:
            completion = client.chat.completions.create(
                model="gemini-1.5-pro", # or appropriate model name
                messages=messages
            )
            # Extract JSON from response by looking for ```json blocks
            content = completion.choices[0].message.content
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                response = json.loads(json_match.group(1))
            else:
                raise ValueError("No JSON block found in model response")
        
        # Ensure all response fields are present
        result = {
            "bottom_line": response.get("bottom_line", ""),
            "additional_insights": response.get("additional_insights", ""),
            "follow_up_questions": response.get("follow_up_questions", []),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "question": request.question,
                "rows_analyzed": len(df),
                "columns_analyzed": len(df.columns)
            }
        }
        
        return result
            
    except Exception as e:
        logging.error(f"Error generating business analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    """Request model for chat history processing
    
    Attributes:
        messages: List of dictionaries containing chat messages
                 Each message must have 'role' and 'content' fields
                 Role must be one of: 'user', 'assistant', 'system'
    """
    messages: List[Dict[str, str]]

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        
        for msg in v:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            if msg['role'] not in ['user', 'assistant', 'system']:
                raise ValueError("Message role must be 'user', 'assistant', or 'system'")
            if not msg['content'].strip():
                raise ValueError("Message content cannot be empty")
        
        return v

async def process_chat(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """Process chat messages and return complete response
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' fields
        
    Returns:
        Dict[str, str]: Dictionary containing response content
        
    Raises:
        Exception: If OpenAI API call fails
    """
    # Convert messages to string format for prompt
    messages_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CHAT},
        {"role": "user", "content": f"Message History:\n{messages_str}"}
    ]

    # Get response based on model mode
    if MODEL_MODE == "openai":
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=prompt_messages,
            response_format={"type": "json_object"}
        )
        response = json.loads(completion.choices[0].message.content)
    elif MODEL_MODE in ["gemini", "anthropic"]:
        completion = client.chat.completions.create(
            model="gemini-1.5-pro", # or appropriate model name
            messages=prompt_messages
        )
        # Extract JSON from response by looking for ```json blocks
        content = completion.choices[0].message.content
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            response = json.loads(json_match.group(1))
        else:
            raise ValueError("No JSON block found in model response")
    
    return response

@app.post("/chat",
    response_model=Dict[str, Any],
    summary="Process chat history",
    description="""
    Process chat history and return enhanced message.
    
    The endpoint:
    - Analyzes conversation context
    - Enhances user messages
    - Maintains conversation coherence
    
    Returns enhanced message response.
    """,
    response_description="Enhanced chat message",
    tags=["Chat Processing"]
)
async def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    Process chat history and return enhanced message.
    
    Parameters:
    - request: ChatRequest containing chat messages
    
    Returns:
    - Dictionary containing enhanced message
    
    Raises:
    - HTTPException: If chat processing fails
    """
    try:
        response = await process_chat(request.messages)
        return response
            
    except Exception as e:
        logging.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@dataclass
class AnalysisResult:
    """Container for analysis results"""
    data: pd.DataFrame
    stdout: str
    stderr: str
    code: str
    execution_time: float
    memory_usage: Dict[str, float]

class CodeValidator:
    """Validates Python code for safety and correctness"""
    ALLOWED_MODULES = {'pandas', 'numpy', 'scipy', 'statsmodels', 'sklearn'}
    
    @staticmethod
    def validate_imports(code: str) -> Tuple[bool, str]:
        """Check if code only imports allowed modules"""
        try:
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend(n.name.split('.')[0] for n in node.names)
                    else:
                        imports.append(node.module.split('.')[0])
                        
            illegal_imports = set(imports) - CodeValidator.ALLOWED_MODULES
            if illegal_imports:
                return False, f"Illegal imports detected: {illegal_imports}"
                
            return True, "Validation passed"
            
        except SyntaxError as e:
            return False, f"Syntax error in code: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

@dataclass
class GeneratedCode:
    """Container for generated analysis code"""
    code: str
    description: str
    estimated_complexity: str
    validation_result: Tuple[bool, str]

@dataclass
class CodeGenerationResult:
    """Container for code generation results"""
    code: str
    description: str
    validation: Dict[str, Any]
    metadata: Dict[str, Any]
    attempts: int
    validation_errors: List[str]

async def generate_analysis_code(
    request: RunAnalysisRequest,
    max_attempts: int = 10
) -> CodeGenerationResult:
    """Generate and validate analysis code with retry logic
    
    Args:
        request: RunAnalysisRequest containing data and question
        max_attempts: Maximum number of retry attempts for validation failures
        
    Returns:
        CodeGenerationResult containing generated code and metadata
    """
    attempts = 0
    validation_errors = []
    
    while attempts < max_attempts:
        attempts += 1
        
        try:
            # Get code from OpenAI
            code_response = await get_python_analysis_code(request)
            
            # Validate the generated code
            is_valid, validation_message = CodeValidator.validate_imports(code_response["code"])
            
            if is_valid:
                return CodeGenerationResult(
                    code=code_response["code"],
                    description=code_response["description"],
                    validation={
                        "is_valid": True,
                        "message": validation_message
                    },
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "question": request.question,
                        "attempts": attempts,
                        "validation_history": validation_errors
                    },
                    attempts=attempts,
                    validation_errors=validation_errors
                )
            
            # If validation failed, add error to history and retry
            validation_errors.append(validation_message)
            
        except Exception as e:
            validation_errors.append(str(e))
            if attempts == max_attempts:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate valid code after {max_attempts} attempts: {str(e)}"
                )

    # If we get here, we've exhausted our attempts
    raise HTTPException(
        status_code=500,
        detail=f"Failed to generate valid code after {max_attempts} attempts"
    )

# Update original endpoint to use new retry logic
@app.post("/run_analysis")
async def run_analysis(request: RunAnalysisRequest) -> Dict[str, Any]:
    """
    Execute complete data analysis workflow with integrated generation and execution retry logic.
    """
    max_attempts = 5  # Single attempt counter for the generate-execute cycle
    attempts = 0
    error_history = []
    
    try:
        # Input validation
        if not request.data:
            raise HTTPException(status_code=422, detail="Input data cannot be empty")
            
        # Convert JSON to DataFrames dictionary
        dataframes = {}
        for dataset_name, dataset_records in request.data.items():
            if dataset_records:
                df = pd.DataFrame(dataset_records)
                dataframes[dataset_name] = df
            else:
                dataframes[dataset_name] = pd.DataFrame()

        while attempts < max_attempts:
            attempts += 1
            
            try:
                # Update request with error context if available
                if error_history:
                    request.error_message = error_history[-1]["error"]
                    request.failed_code = error_history[-1]["code"]

                # Generate code
                code_result = await generate_analysis_code(request)

                # Validate the generated code
                if not code_result.validation["is_valid"]:
                    error_info = {
                        "attempt": attempts,
                        "error": code_result.validation["message"],
                        "code": code_result.code,
                        "timestamp": datetime.now().isoformat(),
                        "type": "validation_error"
                    }
                    error_history.append(error_info)
                    continue

                # Create namespace for execution
                namespace = {
                    'pd': pd,
                    'np': np,
                    'dfs': dataframes
                }

                # Capture stdout and stderr
                stdout = io.StringIO()
                stderr = io.StringIO()

                # Execute the code
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exec(code_result.code, namespace)
                    
                    if 'analyze_data' not in namespace:
                        raise ValueError("Generated code did not define analyze_data function")
                        
                    result = namespace['analyze_data'](dataframes)
                    
                    if not isinstance(result, (pd.DataFrame, list, dict)):
                        result = pd.DataFrame(result)

                # If we get here, execution was successful
                return {
                    "status": "success",
                    "code": code_result.code,
                    "data": result.to_dict('records') if isinstance(result, pd.DataFrame) else result,
                    "metadata": {
                        "execution_time": datetime.now().isoformat(),
                        "attempts": attempts,
                        "error_history": error_history,
                        "execution_details": {
                            "stdout": stdout.getvalue(),
                            "stderr": stderr.getvalue()
                        },
                        "datasets_analyzed": len(dataframes),
                        "total_rows_analyzed": sum(len(df) for df in dataframes.values() if not df.empty),
                        "total_columns_analyzed": sum(len(df.columns) for df in dataframes.values() if not df.empty)
                    }
                }

            except Exception as e:
                # Classify and record the error
                error_info = {
                    "attempt": attempts,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "code": code_result.code if 'code_result' in locals() else None,
                    "stdout": stdout.getvalue() if 'stdout' in locals() else "",
                    "stderr": stderr.getvalue() if 'stderr' in locals() else "",
                    "timestamp": datetime.now().isoformat()
                }
                error_history.append(error_info)

                if attempts >= max_attempts:
                    return {
                        "status": "failed",
                        "error_history": error_history,
                        "last_error": str(e),
                        "suggestions": "Consider reformulating the question or checking data quality"
                    }

    except Exception as e:
        logging.error(f"Error in run_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def is_date_column(series: pd.Series) -> bool:
    """Check if a pandas Series likely contains date values
    
    Args:
        series: pandas Series to check
        
    Returns:
        bool: True if series likely contains dates, False otherwise
    """
    # Skip if series is empty
    if series.empty:
        return False
        
    # Get non-null values
    sample = series.dropna().head(100)
    if sample.empty:
        return False
        
    # Common date patterns
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY or MM-DD-YYYY
        r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
        r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        r'\d{2}\.\d{2}\.\d{4}',  # DD.MM.YYYY or MM.DD.YYYY
        r'\d{4}\.\d{2}\.\d{2}'   # YYYY.MM.DD
    ]
    
    # Check if any values match date patterns
    pattern = '|'.join(date_patterns)
    matches = sample.astype(str).str.match(pattern)
    match_ratio = matches.mean() if not matches.empty else 0
    
    return match_ratio > 0.8  # Return True if >80% of values match date patterns

def convert_to_datetime(value: Any, column: str) -> Optional[datetime]:
    """Convert a value to datetime with flexible format handling
    
    Args:
        value: Value to convert
        column: Column name for error reporting
        
    Returns:
        datetime or None if conversion fails
    """
    if pd.isna(value):
        return None
        
    try:
        # First try pandas to_datetime with coerce
        result = pd.to_datetime(value, infer_datetime_format=True)
        # Convert Timestamp to datetime
        if isinstance(result, pd.Timestamp):
            return result.to_pydatetime()
        return result
    except:
        try:
            # Try dateutil parser as fallback
            from dateutil import parser
            parsed = parser.parse(str(value))
            # Ensure we return a datetime object
            return parsed.replace(tzinfo=None)
        except:
            return None

def convert_datetime_series(series: pd.Series) -> pd.Series:
    """Convert a series of values to datetime using vectorized operations
    
    Args:
        series: pandas Series to convert
        
    Returns:
        pandas Series with ISO format datetime strings
    """
    try:
        # Convert to datetime
        result = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
        # Convert to ISO format strings
        return result.dt.strftime('%Y-%m-%dT%H:%M:%S')
    except Exception as e:
        logging.warning(f"Initial datetime conversion failed: {str(e)}")
        return series

class AnalysisError(Exception):
    def __init__(self, message: str, error_type: str, code: str = None):
        self.message = message
        self.error_type = error_type
        self.code = code
        super().__init__(self.message)

def classify_error(error: Exception, code: str = None) -> AnalysisError:
    """Classify the type of error to inform retry strategy"""
    if isinstance(error, SyntaxError):
        return AnalysisError(str(error), "syntax", code)
    elif isinstance(error, NameError):
        return AnalysisError(str(error), "undefined_variable", code)
    elif isinstance(error, ValueError):
        return AnalysisError(str(error), "value_error", code)
    # ... add more classifications
    return AnalysisError(str(error), "unknown", code)










