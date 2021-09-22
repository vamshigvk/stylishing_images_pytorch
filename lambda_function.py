import json
import boto3

from stylize import validate
print('validate imported from stylize')
print('Loading function')

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    destination_bucket_name = 'csv2dynamodbvialambda'

    # event contains all information about uploaded object
    print("Event :", event)

    # Bucket Name where file was uploaded
    source_bucket_name = event['Records'][0]['s3']['bucket']['name']

    # Filename of object (with path)
    file_key_name = event['Records'][0]['s3']['object']['key']

    # Copy Source Object
    copy_source_object = {'Bucket': source_bucket_name, 'Key': file_key_name}

    # S3 copy object operation
    s3_client.copy_object(CopySource=copy_source_object, Bucket=destination_bucket_name, Key=file_key_name)

    #validate("./test/content","./test/styled")

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from S3 events Lambda!')
    }