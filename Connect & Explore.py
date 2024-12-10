import streamlit as st
import pandas as pd
import os
import warnings
import logging
from typing import Dict, Any, List
import asyncio

# Import FastAPI functions
from dataAnalystAPI import (
    CleanseRequest, 
    DatasetInput,
    cleanse_dataframes,
    get_dictionary,
    DictionaryRequest
)

# Add imports for DataRobot
import datarobot as dr

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.datasets = {}
    st.session_state.cleansed_data = {}
    st.session_state.data_dictionaries = {}

# Modify process_data to handle coroutine reuse
@st.cache_resource(show_spinner=False)
def process_data_cached(datasets_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Wrapper function to handle async processing with caching
    """
    return asyncio.run(process_data_async(datasets_dict))

async def process_data_async(datasets_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Process and cleanse the uploaded datasets using the API functions.
    Returns a dictionary containing cleansed data and success status.
    """
    try:
        # Convert the datasets dictionary to a list of DatasetInput objects
        datasets_list = [
            DatasetInput(
                name=name,
                data=df.to_dict(orient='records')
            )
            for name, df in datasets_dict.items()
        ]
        
        # Create cleanse request with the list
        cleanse_request = CleanseRequest(datasets=datasets_list)
        
        # Cleanse the data
        cleansed_results = await cleanse_dataframes(cleanse_request)
        
        # Format results
        cleansed_data = {
            dataset.name: {
                'data': pd.DataFrame(dataset.data),
                'report': dataset.cleaning_report
            }
            for dataset in cleansed_results.datasets
        }
        
        return {'success': True, 'data': cleansed_data}
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return {'success': False, 'error': str(e)}

# Modify generate_dictionaries similarly
@st.cache_resource(show_spinner=False)
def generate_dictionaries_cached(_cleansed_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Wrapper function to handle async dictionary generation with caching
    """
    return asyncio.run(generate_dictionaries_async(_cleansed_data))

async def generate_dictionaries_async(_cleansed_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate data dictionaries for all datasets"""
    try:
        # Create a list of DatasetInput objects
        datasets = []
        logger.info(f"Starting dictionary generation for {len(_cleansed_data)} datasets")
        
        for name, data in _cleansed_data.items():
            if isinstance(data, dict) and 'data' in data:
                df = data['data']
                datasets.append(
                    DatasetInput(
                        name=name,
                        data=df.to_dict(orient='records')
                    )
                )
                logger.info(f"Added dataset {name} for dictionary generation")

        # Create the request with the datasets
        request_data = CleanseRequest(datasets=datasets)
        
        dictionary_response = await get_dictionary(request_data)
        
        if dictionary_response and isinstance(dictionary_response, dict):
            if 'dictionaries' in dictionary_response:
                result_dict = {
                    dict_entry['name']: dict_entry['dictionary']
                    for dict_entry in dictionary_response['dictionaries']
                    if dict_entry.get('name') and 'dictionary' in dict_entry
                }
                logger.info(f"Successfully generated dictionaries for {len(result_dict)} datasets")
                return result_dict
            else:
                logger.warning("Dictionary response missing 'dictionaries' key")
        else:
            logger.warning(f"Unexpected dictionary response format: {type(dictionary_response)}")
        
        return {}
            
    except Exception as e:
        logger.error(f"Error generating dictionaries: {str(e)}", exc_info=True)
        return {}

@st.cache_data(show_spinner=False)
def process_uploaded_file(file):
    """Process a single uploaded file and return a list of (dataset_name, dataframe) tuples
    
    Args:
        file: The uploaded file object
    Returns:
        list: List of (dataset_name, dataframe) tuples, or empty list if error
    """
    try:
        logger.info(f"Processing uploaded file: {file.name}")
        file_extension = os.path.splitext(file.name)[1].lower()
        results = []
        
        if file_extension == '.csv':
            df = pd.read_csv(file)
            dataset_name = os.path.splitext(file.name)[0]
            results.append((dataset_name, df))
            logger.info(f"Loaded CSV {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
            
        elif file_extension in ['.xlsx', '.xls']:
            # Read all sheets
            excel_file = pd.ExcelFile(file)
            base_name = os.path.splitext(file.name)[0]
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                # Use sheet name as dataset name if multiple sheets, otherwise use file name
                dataset_name = (
                    f"{base_name}_{sheet_name}" 
                    if len(excel_file.sheet_names) > 1 
                    else base_name
                )
                results.append((dataset_name, df))
                logger.info(f"Loaded Excel sheet {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error loading {file.name}: {str(e)}", exc_info=True)
        return []

# Add DataRobot initialization function
@st.cache_resource
def init_datarobot():
    """Initialize DataRobot client connection"""
    try:
        return dr.Client(token=os.getenv("DATAROBOT_API_KEY"), endpoint=os.getenv("DATAROBOT_API_ENDPOINT"))
    except Exception as e:
        st.error(f"Failed to connect to DataRobot: {str(e)}")
        return None

# Add function to get catalog datasets
@st.cache_data(show_spinner=False)
def get_catalog_datasets(limit: int = 100) -> List[Dict]:
    """Fetch datasets from AI Catalog with specified limit"""
    with st.spinner("Listing AI Catalog datasets..."):
        try:
            # Get all datasets and manually limit the results
            datasets = dr.Dataset.list()
            datasets = datasets[:limit]
                
            return [{
                'id': ds.id,
                'name': ds.name,
                'created': ds.creation_date.strftime('%Y-%m-%d') if hasattr(ds, 'creation_date') else 'N/A',
                'size': f"{ds.size / (1024*1024):.1f} MB" if hasattr(ds, 'size') else 'N/A'
            } for ds in datasets]
        except Exception as e:
            logger.error(f"Failed to fetch datasets: {str(e)}")
            return []

# Add function to load selected datasets
@st.cache_data(show_spinner=False) 
def get_datasets_as_df(_dataset_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """Load selected datasets as pandas DataFrames"""
    dataframes = {}
    total = len(_dataset_ids)
    for idx, id in enumerate(_dataset_ids, 1):
        dataset = dr.Dataset.get(id)
        try:
            dataframes[dataset.name] = dataset.get_as_dataframe()
            logger.info(f"Successfully downloaded {dataset.name}")
        except Exception as e:
            logger.error(f"Failed to read dataset {dataset.name}: {str(e)}")
            continue
    return dataframes

# Add callback for AI Catalog dataset selection
def catalog_download_callback():
    """Callback function for AI Catalog dataset download"""
    if 'selected_catalog_datasets' in st.session_state and st.session_state.selected_catalog_datasets:
        with st.sidebar:  # Use sidebar context
            with st.spinner("Loading selected datasets..."):
                selected_ids = [ds['id'] for ds in st.session_state.selected_catalog_datasets]
                dataframes = get_datasets_as_df(selected_ids)
                
                # Add downloaded dataframes to session state
                for name, df in dataframes.items():
                    st.session_state.datasets[name] = df
                
                # Process the new data
                results = process_data_cached(st.session_state.datasets)
                
                if results['success']:
                    st.session_state.cleansed_data = results['data']
                    logger.info("Data processing successful, generating dictionaries")
                    
                    # Generate data dictionaries
                    st.session_state.data_dictionaries = generate_dictionaries_cached(
                        st.session_state.cleansed_data
                    )
                    
                    if st.session_state.data_dictionaries:
                        st.success("✅ Data processed and dictionaries generated successfully!")
                    else:
                        st.warning("⚠️ Data processed but there were issues generating some dictionaries")
                else:
                    logger.error("Data processing failed")
                    st.error(f"❌ Error processing data: {results.get('error', 'Unknown error')}")

def clear_data_callback():
    """Callback function to clear all data from session state and cache"""
    # Clear session state
    st.session_state.datasets = {}
    st.session_state.cleansed_data = {}
    st.session_state.data_dictionaries = {}
    st.session_state.selected_catalog_datasets = []  # Also clear catalog selection
    
    # Clear all Streamlit caches
    st.cache_data.clear()
    st.cache_resource.clear()


# Page config
st.set_page_config(
    page_title="Connect Data",
    page_icon="datarobot icon.svg",
    layout="wide"
)


# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1c83e1;
    }
    .stDownloadButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for data upload and processing
with st.sidebar:
    st.title("Connect")       
    col1, col2, col3 = st.columns([1, 4, 4])
    with col1:
        st.image("csv_File_Logo.svg", width=25)
    with col2:
        st.write("**Load Data Files**")
    uploaded_files = st.file_uploader(
        "Select 1 or multiple files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Loading and processing files..."):
            # Process uploaded files
            for file in uploaded_files:
                dataset_results = process_uploaded_file(file)
                if dataset_results:
                    for dataset_name, df in dataset_results:
                        st.session_state.datasets[dataset_name] = df
                        st.success(f"✓ {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
                else:
                    st.error(f"Error loading {file.name}")
            
            # Process data and generate dictionaries
            logger.info("Starting data processing")
            results = process_data_cached(st.session_state.datasets)
            
            if results['success']:
                st.session_state.cleansed_data = results['data']
                logger.info("Data processing successful, generating dictionaries")
                
                # Generate data dictionaries
                st.session_state.data_dictionaries = generate_dictionaries_cached(
                    st.session_state.cleansed_data
                )
                
                if st.session_state.data_dictionaries:
                    st.success("✅ Data processed and dictionaries generated successfully!")
                    st.info("View the generated data dictionaries in the [Data Dictionary](/Data_Dictionary) page")
                else:
                    st.warning("⚠️ Data processed but there were issues generating some dictionaries")
            else:
                logger.error("Data processing failed")
                st.error(f"❌ Error processing data: {results.get('error', 'Unknown error')}")

     
    # Add AI Catalog section
    st.subheader("☁️   DataRobot AI Catalog")
    
    # Initialize DataRobot client
    client = init_datarobot()
    if client:
        # Get datasets from catalog
        datasets = get_catalog_datasets()
        
        # Create form for dataset selection
        with st.form("catalog_selection_form"):
            selected_catalog_datasets = st.multiselect(
                "Select datasets from AI Catalog",
                options=datasets,
                format_func=lambda x: f"{x['name']} ({x['size']})",
                help="You can select multiple datasets",
                key='selected_catalog_datasets'
            )
            
            # Form submit button - Remove the disabled parameter
            submit_button = st.form_submit_button(
                "Load Selected Datasets", 
                on_click=catalog_download_callback
            )
            
            # Process form submission
            if submit_button and len(selected_catalog_datasets) > 0:
                # The callback will handle the download and processing
                pass
            elif submit_button:
                st.warning("Please select at least one dataset")
    else:
        st.warning("Unable to connect to DataRobot AI Catalog")
    
    # Add clear data button
    st.button(
        "Clear All Data",
        help="Remove all loaded datasets and reset the application",
        use_container_width=False,
        type="secondary",
        on_click=clear_data_callback
    )

# Main content area
st.image("datarobot logo.svg", width=200)
st.title("Explore")

# Main content area - conditional rendering based on cleansed data
if not st.session_state.cleansed_data:
    st.info("Upload and process your data using the sidebar to get started")
else:
    for name, data in st.session_state.cleansed_data.items():
        st.subheader(f"{name}")
        
        # Display cleaning report in expander
        with st.expander("View Cleaning Report"):
            report = data['report']
            if report.columns_cleaned:
                st.write("**Columns Cleaned:**")
                st.write(", ".join(report.columns_cleaned))
                
                if report.warnings:
                    st.write("**Warnings:**")
                    for warning in report.warnings:
                        st.write(f"- {warning}")
                    
                if report.errors:
                    st.error("**Errors:**")
                    for error in report.errors:
                        st.write(f"- {error}")

        # Display dataframe with column filters
        df = data['data']
        
        # Create column filters
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input(
                "Search columns",
                key=f"search_{name}",
                help="Filter columns by name"
            )
        with col2:
            n_rows = st.number_input(
                "Rows to display",
                min_value=1,
                max_value=len(df),
                value=min(10, len(df)),
                key=f"n_rows_{name}"
            )
        
        # Filter columns based on search
        if search:
            cols = [col for col in df.columns if search.lower() in col.lower()]
        else:
            cols = df.columns
        
        # Display filtered dataframe
        st.dataframe(
            df[cols].head(n_rows),
            use_container_width=True
        )
        
        # Download button
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Cleansed Data",
                data=csv,
                file_name=f"{name}_cleansed.csv",
                mime="text/csv",
                key=f"download_{name}"
            )

        st.markdown("---")

