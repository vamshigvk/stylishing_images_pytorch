# stylishing_images_pytorch
When you upload an image to the AWS S3 bucket, an event will trigger the AWS Lambda function. AWS Lambda function will be deployed using a Docker image which consists of LambdaHandler function and ML/DL application. The uploaded image will be stored in a temporary folder inside the docker container. ML/DL application will take the uploaded image(s) as input and produces another set of images in an output folder. Finally, the images from the output folder are uploaded to another S3 bucket.

Working on AWS, takes stylish as input and produces model file .pth as output.
Stlish images are fed, retraining the models is done and used at a later point. 

#Docker
#S3
#Lambda
#Python
