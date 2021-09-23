import boto3
from io import StringIO, BytesIO

access_key = 'AKIAWXYSJZ5ZAQ6I3IMD'
secret_access='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'
region_name='us-east-1'

s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_acess_key = secret_access)

bucket_name = "csv2dynamodbvialambda"

s3 = boto3.client('s3')
s3.download_file(bucket_name, "trees.csv", "test/content/"+"tree1.csv")
s3.upload_file( "test/content/"+"tree1.csv", bucket_name, 'tree_upload.csv')


