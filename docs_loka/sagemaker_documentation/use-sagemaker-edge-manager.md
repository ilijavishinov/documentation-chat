# Use Amazon SageMaker Edge Manager on Greengrass core devices<a name="use-sagemaker-edge-manager"></a>

Amazon SageMaker Edge Manager is a software agent that runs on edge devices\. SageMaker Edge Manager provides model management for edge devices so that you can package and use Amazon SageMaker Neo\-compiled models directly on Greengrass core devices\. By using SageMaker Edge Manager, you can also sample model input and output data from your core devices, and send that data to the AWS Cloud for monitoring and analysis\. Because SageMaker Edge Manager uses SageMaker Neo to optimize your models for your target hardware, you don't need to install the DLR runtime directly on your device\. On Greengrass devices, SageMaker Edge Manager doesn't load local AWS IoT certificates or call the AWS IoT credential provider endpoint directly\. Instead, SageMaker Edge Manager uses the [token exchange service](token-exchange-service-component.md) to fetch temporary credential from a TES endpoint\. 

This section describes how SageMaker Edge Manager works on Greengrass core devices\.



## How SageMaker Edge Manager works on Greengrass devices<a name="how-to-use-sdge-manager-with-greengrass"></a>

To deploy the SageMaker Edge Manager agent to your core devices, create a deployment that includes the `aws.greengrass.SageMakerEdgeManager` component\. AWS IoT Greengrass manages the installation and lifecycle of the Edge Manager agent on your devices\. When a new version of the agent binary is available, deploy the updated version of the `aws.greengrass.SageMakerEdgeManager` component to upgrade the version of the agent that is installed on your device\. 

When you use SageMaker Edge Manager with AWS IoT Greengrass, your workflow includes the following high\-level steps:

1. Compile models with SageMaker Neo\.

1. Package your SageMaker Neo\-compiled models using SageMaker edge packaging jobs\. When you run an edge packaging job for your model, you can choose to create a model component with the packaged model as an artifact that can be deployed to your Greengrass core device\. 

1. Create a custom inference component\. You use this inference component to interact with the Edge Manager agent to perform inference on the core device\. These operations include loading models, invoke prediction requests to run inference, and unloading models when the component shuts down\. 

1. Deploy the SageMaker Edge Manager component, the packaged model component, and the inference component to run your model on the SageMaker inference engine \(Edge Manager agent\) on your device\.

For more information about creating edge packaging jobs and inference components that work with SageMaker Edge Manager, see [Deploy Model Package and Edge Manager Agent with AWS IoT Greengrass](https://docs.aws.amazon.com/sagemaker/latest/dg/edge-greengrass.html) in the *Amazon SageMaker Developer Guide*\.

The [Tutorial: Get started with SageMaker Edge Manager](get-started-with-edge-manager-on-greengrass.md) tutorial shows you how to set up and use the SageMaker Edge Manager agent on an existing Greengrass core device, using AWS\-provided example code that you can use to create sample inference and model components\. 

When you use SageMaker Edge Manager on Greengrass core devices, you can also use the capture data feature to upload sample data to the AWS Cloud\. Capture data is a SageMaker feature that you use to upload inference input, inference results, and additional inference data to an S3 bucket or a local directory for future analysis\. For more information about using capture data with SageMaker Edge Manager, see [Manage Model](https://docs.aws.amazon.com/sagemaker/latest/dg/edge-manage-model.html#edge-manage-model-capturedata) in the *Amazon SageMaker Developer Guide*\.

## Requirements<a name="greengrass-edge-manager-agent-requirements"></a>

You must meet the following requirements to use the SageMaker Edge Manager agent on Greengrass core devices\.<a name="sm-edge-manager-component-reqs"></a>
+ <a name="sm-req-core-device"></a>A Greengrass core device running on Amazon Linux 2, a Debian\-based Linux platform \(x86\_64 or Armv8\), or Windows \(x86\_64\)\. If you don't have one, see [Tutorial: Getting started with AWS IoT Greengrass V2](getting-started.md)\.
+ <a name="sm-req-python"></a>[Python](https://www.python.org/downloads/) 3\.6 or later, including `pip` for your version of Python, installed on your core device\.
+ The [Greengrass device role](device-service-role.md) configured with the following: 
  + <a name="sm-req-iam-trust-relationship"></a>A trust relationship that allows `credentials.iot.amazonaws.com` and `sagemaker.amazonaws.com` to assume the role, as shown in the following IAM policy example\.

    ```
    { 
      "Version": "2012-10-17",
      "Statement": [ 
        { 
          "Effect": "Allow", 
          "Principal": {
            "Service": "credentials.iot.amazonaws.com"
           }, 
          "Action": "sts:AssumeRole" 
        },
        { 
          "Effect": "Allow", 
          "Principal": {
            "Service": "sagemaker.amazonaws.com"
          }, 
          "Action": "sts:AssumeRole" 
        } 
      ] 
    }
    ```
  + <a name="sm-req-iam-sagemanakeredgedevicefleetpolicy"></a>The [AmazonSageMakerEdgeDeviceFleetPolicy](https://console.aws.amazon.com/iam/home#/policies/arn:aws:iam::aws:policy/service-role/AmazonSageMakerEdgeDeviceFleetPolicy) IAM managed policy\.
  + <a name="sm-req-iam-s3-putobject"></a>The `s3:PutObject` action, as shown in the following IAM policy example\.

    ```
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Action": [
            "s3:PutObject"
          ],
          "Resource": [
            "*"
          ],
          "Effect": "Allow"
        }
      ]
    }
    ```
+ <a name="sm-req-s3-bucket"></a>An Amazon S3 bucket created in the same AWS account and AWS Region as your Greengrass core device\. SageMaker Edge Manager requires an S3 bucket to create an edge device fleet, and to store sample data from running inference on your device\. For information about creating S3 buckets, see [Getting started with Amazon S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html)\.
+ <a name="sm-req-edge-device-fleet"></a>A SageMaker edge device fleet that uses the same AWS IoT role alias as your Greengrass core device\. For more information, see [Create an edge device fleet](get-started-with-edge-manager-on-greengrass.md#create-edge-device-fleet-for-greengrass)\.
+ <a name="sm-req-edge-device"></a>Your Greengrass core device registered as an edge device in your SageMaker Edge device fleet\. The edge device name must match the AWS IoT thing name for your core device\. For more information, see [Register your Greengrass core device](get-started-with-edge-manager-on-greengrass.md#register-greengrass-core-device-in-sme)\.

## Get started with SageMaker Edge Manager<a name="use-sm-edge-manager"></a>

You can complete a tutorial to get started using SageMaker Edge Manager\. The tutorial shows you how to get started using SageMaker Edge Manager with AWS\-provided sample components on an existing core device\. These sample components use the SageMaker Edge Manager component as a dependency to deploy the Edge Manager agent, and perform inference using pre\-trained models that were compiled using SageMaker Neo\. For more information, see [Tutorial: Get started with SageMaker Edge Manager](get-started-with-edge-manager-on-greengrass.md)\.