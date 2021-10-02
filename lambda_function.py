import json
import logging
import os
import time
import uuid
from datetime import datetime
import boto3

s3_resource = boto3.resource('s3')
ecs = boto3.client('ecs')

def lambda_handler(event, context):

    # event contains all information about uploaded object
    print("Event :", event)

    # Bucket Name where file was uploaded
    SOURCE_BUCKET_NAME = event['Records'][0]['s3']['bucket']['name']
    print("Source bucket name is: ", SOURCE_BUCKET_NAME, "only")

    # Filename of object (with path)
    FILE_KEY_NAME = event['Records'][0]['s3']['object']['key']
    print('File key name is: ', FILE_KEY_NAME, " and bucket name is: ",SOURCE_BUCKET_NAME," only")

    print('Getting environment variables')
    DESTINATION_BUCKET_NAME = os.environ['DESTINATION_BUCKET_NAME']
    FARGATE_CLUSTER = os.environ['FARGATE_CLUSTER']
    FARGATE_TASK_DEF_NAME = os.environ['FARGATE_TASK_DEF_NAME']
    FARGATE_SUBNET_ID = os.environ['FARGATE_SUBNET_ID']
    DOCKERHUB_REPO =  os.environ['DOCKERHUB_REPO']
    print('Got these environment variables:',DESTINATION_BUCKET_NAME,FARGATE_CLUSTER,FARGATE_TASK_DEF_NAME,FARGATE_SUBNET_ID,DOCKERHUB_REPO)

    print('triggering ECS RUN TASK')
    response = ecs.run_task(
            cluster=FARGATE_CLUSTER,
            launchType = 'FARGATE',
            taskDefinition=FARGATE_TASK_DEF_NAME,
            count = 1,
            platformVersion='LATEST',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': [
                        FARGATE_SUBNET_ID,
                    ],
                    'assignPublicIp': 'ENABLED'
                }
            },
            overrides={
                'containerOverrides': [
                    {
                        'name': 'stylieze_retraining',
                        'environment': [
                            {
                                'name': 'DESTINATION_BUCKET_NAME',
                                'value': DESTINATION_BUCKET_NAME
                            },
                            {
                                'name': 'SOURCE_BUCKET_NAME',
                                'value': SOURCE_BUCKET_NAME
                            },
                            {
                                'name': 'FILE_KEY_NAME',
                                'value': FILE_KEY_NAME
                            },
                        ],
                    },
                ],
            },
        )

    return {
        'statusCode': 200,
        'body': json.dumps('Fetched filename and triggered run task', response)
    }

 