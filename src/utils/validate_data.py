# Data Quality/Validation with Great Expectations
# make sure your data meets certain expectations before processing it further
import great_expectations as ge
from typing import Tuple, List

def validate_data(df) -> Tuple[bool, List[str]]:

    """Implement critical data quality checks  """
    print("Validating data quality...")
    ge_df = ge.dataset.PandasDataset(df)

    # validating schema and required columns

    # customer must exist
    ge_df.expect_column_to_exist('customerID')
    ge_df.expect_column_values_to_not_be_null('customerID')

    ge_df.expect_column_to_exist('gender')
    ge_df.expect_column_to_exist('country')
    ge_df.expect_column_to_exist('age')
    ge_df.expect_column_to_exist('tenure')

    # validate inputs
    ge_df.expect_column_values_to_be_in_set('churn', ['Yes', 'No'])
    ge_df.expect_column_values_to_be_in_set('gender', ['Male','Female'])
    ge_df.expect_column_values_to_be_in_set('country', ['Germany', 'France', 'Spain'])


    # numerical input validation
    ge_df.expect_column_values_to_be_between('tenure', min_value=0)
    ge_df.expect_column_values_to_be_between('age', min_value=0, max_value=120)

    ge_df.expect_column_values_to_be_between('credit_score', min_value=300, max_value=850)
    ge_df.expect_column_values_to_be_between('credit_card', min_value=0)

    pass