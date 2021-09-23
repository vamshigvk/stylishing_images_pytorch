import json
import boto3

from stylize import validate
print('validate imported from stylize')
print('Loading function')

s3 = boto3.client('s3')
#s3_resource = boto3.resource('s3')

def lambda_handler(event, context):
    destination_bucket_name = 'csv2dynamodbvialambda'

    # event contains all information about uploaded object
    print("Event :", event)

    # Bucket Name where file was uploaded
    source_bucket_name = event['Records'][0]['s3']['bucket']['name']

    # Filename of object (with path)
    file_key_name = event['Records'][0]['s3']['object']['key']

    #Copy Source Object
    #copy_source_object = {'Bucket': source_bucket_name, 'Key': file_key_name}

    #Downloading newly added object to test/content/ folder
    s3.download_file(source_bucket_name, file_key_name, "test/content/"+file_key_name)
    print("downloaded a new image with file name: ",file_key_name," bucket name: ", source_bucket_name)

    # S3 copy object operation
    #s3_client.copy_object(CopySource=copy_source_object, Bucket=destination_bucket_name, Key=file_key_name)
    
    #Validates the downloaded image from S3 source bucket and stores the output image at test/styled
    print('validate is about to trigger')
    validate("./test/content","./test/styled")
    print("validate is finished and stylized the image")

    style_list = ['bayanihan','lazy','mosaic','starry','tokyo_ghoul','udnie','wave']
    for i in len(range(style_list)):
        s3.upload_file( "test/content/"+style_list[i]+file_key_name, destination_bucket_name, "stylized"+file_key_name)
        print('Uploaded file from style: ',style_list[i])

    print('uploaded all the files to destination bucket')

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from S3 events Lambda!')
    }