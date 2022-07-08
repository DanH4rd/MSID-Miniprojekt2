import json
from dataprofiler import Data, Profiler

data = Data("Assessor_Historical_Secured_Property_Tax_Rolls.csv") # Auto-Detect & Load: CSV, AVRO, Parquet, JSON, Text
print(data.data.head(5)) # Access data directly via a compatible Pandas DataFrame

profile = Profiler(data) # Calculate Statistics, Entity Recognition, etc
readable_report = profile.report(report_options={"output_format":"pretty"})
print(json.dumps(readable_report, indent=4))
