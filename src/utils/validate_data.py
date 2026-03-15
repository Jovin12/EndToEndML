import great_expectations as gx
from typing import Tuple, List
import pandas as pd

def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Data quality checks for GX 1.15.1+.
    """
    print("Validating data quality...")

    # 1. Get the context (Ephemeral for in-memory)
    context = gx.get_context()

    # 2. Add or update the Pandas Data Source
    # Note: Use 'data_sources' (plural, with underscore)
    datasource_name = "my_pandas_datasource"
    datasource = context.data_sources.add_or_update_pandas(name=datasource_name)
    
    # 3. Add a Data Asset and create a Batch Definition
    asset_name = "churn_data_asset"
    data_asset = datasource.add_dataframe_asset(name=asset_name)
    
    # Use 'whole_dataframe' definition for simple in-memory validation
    batch_definition = data_asset.add_batch_definition_whole_dataframe("all_data")

    # 4. Create an Expectation Suite
    suite_name = "churn_data_suite"
    suite = gx.ExpectationSuite(name=suite_name)
    
    # Add expectations to the suite object
    # suite.add_expectation(gx.expectations.ExpectColumnToExist(column="customerID"))
    # suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="customerID"))
    # suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="churn", value_set=["Yes", "No"]))
    # suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]))
    # suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0))
    # suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="age", min_value=0, max_value=120))

    # Add the suite to the context
    context.suites.add(suite)

    # 5. Create a Validation Definition (Connects Data + Suite)
    validation_definition = gx.ValidationDefinition(
        name="churn_validation_def",
        data=batch_definition,
        suite=suite
    )
    context.validation_definitions.add(validation_definition)

    # 6. Run validation
    # Pass the dataframe specifically into the run command
    results = validation_definition.run(batch_parameters={"dataframe": df})

    # 7. Parse results
    failed_expectations = []
    for result in results.results:
        if not result.success:
            # Safely extract failure details
            exp_type = result.expectation_config.type
            column = result.expectation_config.kwargs.get("column", "N/A")
            failed_expectations.append(f"{exp_type} on column '{column}' failed")

    is_valid = results.success

    return is_valid, failed_expectations