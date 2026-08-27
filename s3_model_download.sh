# modify the local location if needed
aws s3 sync s3://amz-s3-receipt-understanding-423051206837-eu-west-3-an/models/vision/ ./services/vision-service/models/
aws s3 sync s3://amz-s3-receipt-understanding-423051206837-eu-west-3-an/models/kie/ ./services/kie-service/models/