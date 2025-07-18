import boto3
import time
import pandas as pd

s3 = boto3.client('s3')


# Upload on S3
bucket_name = 'ucwdc-country-classifier'
file_path = './textract/UCWDC-Competition-Music-Jun-2019-Public-List.pdf'
object_name = 'UCWDC-Competition-Music-Jun-2019-Public-List.pdf'  # S3 key


# ## Upload PDF
# s3.upload_file(file_path, bucket_name, object_name)
# print(f"Uploaded {file_path} to s3://{bucket_name}/{object_name}")

# # # Add to Textract
textract = boto3.client('textract')

# ## Start the async job
# response = textract.start_document_analysis(
#     DocumentLocation={
#         'S3Object': {
#             'Bucket': bucket_name,
#             'Name': object_name
#         }
#     },
#     FeatureTypes=["TABLES",'FORMS']
# )

# job_id = response['JobId']
# print(f"Started Textract job with ID: {job_id}")

## see status
# while True:
#     result = textract.get_document_analysis(JobId=job_id)
#     status = result['JobStatus']
#     print(f"Job status: {status}")
#     if status in ['SUCCEEDED', 'FAILED']:
#         break
#     time.sleep(5)

job_id = "75e23eacbea1ab1eca461d2d4ec063145607d00590a73243977dd6267179655c"

all_blocks = []
next_token = None

# Paginate through all results
while True:
    if next_token:
        response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
    else:
        response = textract.get_document_analysis(JobId=job_id)

    all_blocks.extend(response['Blocks'])

    next_token = response.get('NextToken')
    if not next_token:
        break

blocks = all_blocks

# Collect table blocks
tables = [block for block in blocks if block['BlockType'] == 'TABLE']

all_rows = []
header_saved = False  # Track if header is already added

# Extract text for each table and accumulate rows
for table_index, table in enumerate(tables):
    # Find child cell IDs
    cell_ids = table['Relationships'][0]['Ids'] if 'Relationships' in table else []

    # Extract cells
    cells = [block for block in blocks if block['Id'] in cell_ids and block['BlockType'] == 'CELL']

    # Group cells by row and column
    rows = {}
    for cell in cells:
        row_index = cell['RowIndex']
        col_index = cell['ColumnIndex']
        text = ''
        # Get text inside the cell
        if 'Relationships' in cell:
            word_ids = [rel_id for rel_id in cell['Relationships'][0]['Ids']]
            words = [block['Text'] for block in blocks if block['Id'] in word_ids and block['BlockType'] == 'WORD']
            text = ' '.join(words)
        rows.setdefault(row_index, {})[col_index] = text

    # Convert rows dict to list of lists ordered by row and col
    max_col = max((max(cols.keys()) for cols in rows.values()), default=0)
    sorted_row_nums = sorted(rows.keys())
    
    if not header_saved:
        # Append header row first (row 1)
        header_row = [rows[sorted_row_nums[0]].get(col, '') for col in range(1, max_col + 1)]
        all_rows.append(header_row)
        header_saved = True
        start_row = 1  # skip header row in the loop below for first table
    else:
        start_row = 1  # skip header row for subsequent tables

    for row_num in sorted_row_nums[start_row:]:
        row_cells = rows[row_num]
        row_list = [row_cells.get(col, '') for col in range(1, max_col + 1)]
        all_rows.append(row_list)

# Create pandas DataFrame with all rows combined
df = pd.DataFrame(all_rows)

# Export to CSV
file_path = './combined_tables.csv'
df.to_csv(file_path, index=False, header=False)

print(f"Exported combined {len(tables)} tables (with headers) to combined_tables.csv")

#added to S3
object_name = 'combined_tables.csv'  # S3 
s3.upload_file(file_path, bucket_name, object_name)
print("Exported to S3")