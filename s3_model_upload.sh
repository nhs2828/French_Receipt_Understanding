# modify the local location if needed
aws s3 sync ./services/vision-service/models/ s3://amz-s3-receipt-understanding-423051206837-eu-west-3-an/models/vision/
aws s3 sync ./services/kie-service/models/ s3://amz-s3-receipt-understanding-423051206837-eu-west-3-an/models/kie/