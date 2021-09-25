import json
import boto3
import os
import botocore

from stylize import validate
print('validate imported from stylize')
print('Loading function')

s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')


def lambda_handler(event, context):

    # event contains all information about uploaded object
    print("Event :", event)

    # Bucket Name where file was uploaded
    source_bucket_name = event['Records'][0]['s3']['bucket']['name']
    print("Source bucket name is: ", source_bucket_name, "only")

    # Filename of object (with path)
    file_key_name = event['Records'][0]['s3']['object']['key']
    print('File key name is: ', file_key_name, "only")

    bucket = s3_resource.Bucket(source_bucket_name)

    try:
        path, filename = os.path.split(file_key_name)
        print('Key we are downloading is: ',filename)
        bucket.download_file(file_key_name, "/tmp/" + filename)
    except:
            print(f"Error occurred while downloading, The object {file_key_name} does not exist")


    #Validates the downloaded image from S3 source bucket and stores the output image at tmp/styled
    print('validate is about to trigger')
    validate("/tmp","/tmp/styled")
    print("validate is finished and stylized the image")

    destination_bucket_name = event['Records'][0]['s3']['bucket']['name']
    print("Destination bucket name is: ", destination_bucket_name, "only")

    style_list = ['bayanihan','lazy','mosaic','starry','tokyo_ghoul','udnie','wave']

    for i in range(len(style_list)):
        #try:
        s3.upload_file( "/tmp/styled/"+style_list[i]+'/'+filename, destination_bucket_name, 'test/styled/'+style_list[i]+'/'+filename)
        print('Uploaded file from style: ',style_list[i],'file uploaded is: ',filename )
        #except :
            #print("Error occurred while uploading the file, ", filename)

    print('uploaded all the files to destination bucket')

    return {
        'statusCode': 200,
        'body': json.dumps('Job done, styled the images from S3 bucket!')
    }