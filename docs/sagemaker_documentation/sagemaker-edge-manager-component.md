# SageMaker Edge Manager<a name="sagemaker-edge-manager-component"></a>

The Amazon SageMaker Edge Manager component \(`aws.greengrass.SageMakerEdgeManager`\) installs the SageMaker Edge Manager agent binary\. 

SageMaker Edge Manager provides model management for edge devices so you can optimize, secure, monitor, and maintain machine learning models on fleets of edge devices\. The SageMaker Edge Manager component installs and manages the lifecycle of the SageMaker Edge Manager agent on your core device\. You can also use SageMaker Edge Manager to package and use SageMaker Neo\-compiled models as model components on Greengrass core devices\. For more information about using SageMaker Edge Manager agent on your core device, see [Use Amazon SageMaker Edge Manager on Greengrass core devices](use-sagemaker-edge-manager.md)\.

SageMaker Edge Manager component v1\.3\.x installs Edge Manager agent binary v1\.20220822\.836f3023\. For more information about Edge Manager agent binary versions, see [Edge Manager Agent](https://docs.aws.amazon.com/sagemaker/latest/dg/edge-device-fleet-about)\.

**Note**  
The SageMaker Edge Manager component is available only in the following AWS Regions:  
US East \(Ohio\)
US East \(N\. Virginia\)
US West \(Oregon\)
EU \(Frankfurt\)
EU \(Ireland\)
Asia Pacific \(Tokyo\)

**Topics**
+ [Versions](#sagemaker-edge-manager-component-versions)
+ [Type](#sagemaker-edge-manager-component-type)
+ [Operating system](#sagemaker-edge-manager-component-os-support)
+ [Requirements](#sagemaker-edge-manager-component-requirements)
+ [Dependencies](#sagemaker-edge-manager-component-dependencies)
+ [Configuration](#sagemaker-edge-manager-component-configuration)
+ [Local log file](#sagemaker-edge-manager-component-log-file)
+ [Changelog](#sagemaker-edge-manager-component-changelog)

## Versions<a name="sagemaker-edge-manager-component-versions"></a>

This component has the following versions:
+ 1\.3\.x
+ 1\.2\.x
+ 1\.1\.x
+ 1\.0\.x

## Type<a name="sagemaker-edge-manager-component-type"></a>

<a name="public-component-type-generic"></a>This <a name="public-component-type-generic-phrase"></a>component is a generic component \(`aws.greengrass.generic`\)\. The [Greengrass nucleus](greengrass-nucleus-component.md) runs the component's lifecycle scripts\.

<a name="public-component-type-more-information"></a>For more information, see [Component types](develop-greengrass-components.md#component-types)\.

## Operating system<a name="sagemaker-edge-manager-component-os-support"></a>

This component can be installed on core devices that run the following operating systems:
+ Linux
+ Windows

## Requirements<a name="sagemaker-edge-manager-component-requirements"></a>

This component has the following requirements:<a name="sm-edge-manager-component-reqs"></a>
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

### Endpoints and ports<a name="sagemaker-edge-manager-component-endpoints"></a>

This component must be able to perform outbound requests to the following endpoints and ports, in addition to endpoints and ports required for basic operation\. For more information, see [Allow device traffic through a proxy or firewall](allow-device-traffic.md)\.


| Endpoint | Port | Required | Description | 
| --- | --- | --- | --- | 
|  `edge.sagemaker.region.amazonaws.com`  | 443 | Yes |  Check device registration status and send metrics to SageMaker\.  | 
|  `*.s3.amazonaws.com`  | 443 | Yes |  Upload capture data to the S3 bucket that you specify\. You can replace `*` with the name of each bucket where you upload data\.  | 

## Dependencies<a name="sagemaker-edge-manager-component-dependencies"></a>

When you deploy a component, AWS IoT Greengrass also deploys compatible versions of its dependencies\. This means that you must meet the requirements for the component and all of its dependencies to successfully deploy the component\. This section lists the dependencies for the [released versions](#sagemaker-edge-manager-component-changelog) of this component and the semantic version constraints that define the component versions for each dependency\. You can also view the dependencies for each version of the component in the [AWS IoT Greengrass console](https://console.aws.amazon.com/greengrass)\. On the component details page, look for the **Dependencies** list\.

------
#### [ 1\.3\.2 ]

The following table lists the dependencies for version 1\.3\.2 of this component\.


| Dependency | Compatible versions | Dependency type | 
| --- | --- | --- | 
| [Greengrass nucleus](greengrass-nucleus-component.md) | >=2\.0\.0 <3\.0\.0 | Soft | 
| [Token exchange service](token-exchange-service-component.md) | >=0\.0\.0 | Hard | 

------
#### [ 1\.3\.1 ]

The following table lists the dependencies for version 1\.3\.1 of this component\.


| Dependency | Compatible versions | Dependency type | 
| --- | --- | --- | 
| [Greengrass nucleus](greengrass-nucleus-component.md) | >=2\.0\.0 <2\.9\.0 | Soft | 
| [Token exchange service](token-exchange-service-component.md) | >=0\.0\.0 | Hard | 

------
#### [ 1\.1\.1 \- 1\.3\.0 ]

The following table lists the dependencies for versions 1\.1\.1 \- 1\.3\.0 of this component\.


| Dependency | Compatible versions | Dependency type | 
| --- | --- | --- | 
| [Greengrass nucleus](greengrass-nucleus-component.md) | >=2\.0\.0 <2\.8\.0 | Soft | 
| [Token exchange service](token-exchange-service-component.md) | >=0\.0\.0 | Hard | 

------
#### [ 1\.1\.0 ]

The following table lists the dependencies for version 1\.1\.0 of this component\.


| Dependency | Compatible versions | Dependency type | 
| --- | --- | --- | 
| [Greengrass nucleus](greengrass-nucleus-component.md) | >=2\.0\.0 <2\.6\.0 | Soft | 
| [Token exchange service](token-exchange-service-component.md) | >=0\.0\.0 | Hard | 

------
#### [ 1\.0\.3 ]

The following table lists the dependencies for version 1\.0\.3 of this component\.


| Dependency | Compatible versions | Dependency type | 
| --- | --- | --- | 
| [Greengrass nucleus](greengrass-nucleus-component.md) | >=2\.0\.0 <2\.5\.0 | Soft | 
| [Token exchange service](token-exchange-service-component.md) | >=0\.0\.0 | Hard | 

------
#### [ 1\.0\.1 and 1\.0\.2 ]

The following table lists the dependencies for versions 1\.0\.1 and 1\.0\.2 of this component\.


| Dependency | Compatible versions | Dependency type | 
| --- | --- | --- | 
| [Greengrass nucleus](greengrass-nucleus-component.md) | >=2\.0\.0 <2\.4\.0 | Soft | 
| [Token exchange service](token-exchange-service-component.md) | >=0\.0\.0 | Hard | 

------
#### [ 1\.0\.0 ]

The following table lists the dependencies for version 1\.0\.0 of this component\.


| Dependency | Compatible versions | Dependency type | 
| --- | --- | --- | 
| [Greengrass nucleus](greengrass-nucleus-component.md) | >=2\.0\.0 <2\.3\.0 | Soft | 
| [Token exchange service](token-exchange-service-component.md) | >=0\.0\.0 | Hard | 

------

For more information about component dependencies, see the [component recipe reference](component-recipe-reference.md#recipe-reference-component-dependencies)\.

## Configuration<a name="sagemaker-edge-manager-component-configuration"></a>

This component provides the following configuration parameters that you can customize when you deploy the component\.

**Note**  
This section describes the configuration parameters that you set in the component\. For more information about the corresponding SageMaker Edge Manager configuration, see [Edge Manager Agent](https://docs.aws.amazon.com/sagemaker/latest/dg/edge-device-fleet-about.html#edge-device-fleet-running-agent) in the *Amazon SageMaker Developer Guide*\.

`DeviceFleetName`  
The name of the SageMaker Edge Manager device fleet that contains your Greengrass core device\.   
You must specify a value for this parameter in the configuration update when you deploy this component\.

`BucketName`  
The name of the S3 bucket to which you upload captured inference data\. The bucket name must contain the string `sagemaker`\.   
If you set `CaptureDataDestination` to `Cloud`, or if you set `CaptureDataPeriodicUpload` to `true`, then you must specify a value for this parameter in the configuration update when you deploy this component\.  
Capture data is an SageMaker feature that you use to upload inference input, inference results, and additional inference data to an S3 bucket or a local directory for future analysis\. For more information about using capture data with SageMaker Edge Manager, see [Manage Model](https://docs.aws.amazon.com/sagemaker/latest/dg/edge-manage-model.html#edge-manage-model-capturedata) in the *Amazon SageMaker Developer Guide*\.

`CaptureDataBatchSize`  
\(Optional\) The size of a batch of capture data requests that the agent handles\. This value must be less than the buffer size that you specify in `CaptureDataBufferSize`\. We recommend that you don't exceed half the buffer size\.  
The agent handles a request batch when the number of requests in the buffer meets the `CaptureDataBatchSize` number, or when the `CaptureDataPushPeriodSeconds` interval elapses, whichever occurs first\.  
Default: `10`

`CaptureDataBufferSize`  
\(Optional\) The maximum number of capture data requests stored in the buffer\.  
Default: `30`

`CaptureDataDestination`  
\(Optional\) The destination where you store captured data\. This parameter can have the following values:  
+ `Cloud`—Uploads captured data to the S3 bucket that you specify in `BucketName`\.
+ `Disk`—Writes captured data to the component's work directory\. 
If you specify `Disk`, you can also choose to periodically upload the captured data to your S3 bucket by setting `CaptureDataPeriodicUpload` to `true`\.  
Default: `Cloud`

`CaptureDataPeriodicUpload`  
\(Optional\) String value that specifies whether to periodically upload captured data\. Supported values are `true` and `false`\.  
Set this parameter to `true` if you set `CaptureDataDestination` to `Disk`, and you also want the agent to periodically upload the captured data your S3 bucket\.  
Default: `false`

`CaptureDataPeriodicUploadPeriodSeconds`  
\(Optional\) The interval in seconds at which SageMaker Edge Manager agent uploads captured data to the S3 bucket\. Use this parameter if you set `CaptureDataPeriodicUpload` to `true`\.  
Default: `8`

`CaptureDataPushPeriodSeconds`  
\(Optional\) The interval in seconds at which SageMaker Edge Manager agent handles a batch of capture data requests from the buffer\.   
The agent handles a request batch when the number of requests in the buffer meets the `CaptureDataBatchSize` number, or when the `CaptureDataPushPeriodSeconds` interval elapses, whichever occurs first\.  
Default: `4`

`CaptureDataBase64EmbedLimit`  
\(Optional\) The maximum size in bytes of captured data that SageMaker Edge Manager agent uploads\.  
Default: `3072`

`FolderPrefix`  
\(Optional\) The name of the folder to which the agent writes the captured data\. If you set `CaptureDataDestination` to `Disk`, the agent creates the folder in the directory that is specified by `CaptureDataDiskPath`\. If you set `CaptureDataDestination` to `Cloud`, or if you set `CaptureDataPeriodicUpload` to `true`, the agent creates the folder in your S3 bucket\.   
Default: `sme-capture`

`CaptureDataDiskPath`  
This feature is available in v1\.1\.0 and later versions of the SageMaker Edge Manager component\.  
\(Optional\) The path to the folder to which the agent creates the captured data folder\. If you set `CaptureDataDestination` to `Disk`, the agent creates the captured data folder in this directory\. If you don't specify this value, the agent creates the captured data folder in the component's work directory\. Use the `FolderPrefix` parameter to specify the name of the captured data folder\.  
Default: `/greengrass/v2/work/aws.greengrass.SageMakerEdgeManager/capture`

`LocalDataRootPath`  
This feature is available in v1\.2\.0 and later versions of the SageMaker Edge Manager component\.  
\(Optional\) The path where this component stores the following data on the core device:  
+ The local database for runtime data when you set `DbEnable` to `true`\.
+ SageMaker Neo\-compiled models that this component automatically downloads when you set `DeploymentEnable` to `true`\.
Default: `/greengrass/v2/work/aws.greengrass.SageMakerEdgeManager`

`DbEnable`  
\(Optional\) You can enable this component to store runtime data in a local database to preserve the data, in case the component fails or the device loses power\.  
This database requires 5 MB of storage on the core device's file system\.  
Default: `false`

`DeploymentEnable`  
This feature is available in v1\.2\.0 and later versions of the SageMaker Edge Manager component\.  
\(Optional\) You can enable this component to automatically retrieve SageMaker Neo\-compiled models from that you upload to Amazon S3\. After you upload a new model to Amazon S3, use SageMaker Studio or the SageMaker API to deploy the new model to this core device\. When you enable this feature, you can deploy new models to core devices without needing to create a AWS IoT Greengrass deployment\.  
To use this feature, you must set `DbEnable` to `true`\. This feature uses the local database to track models that it retrieves from the AWS Cloud\.
Default: `false`

`DeploymentPollInterval`  
This feature is available in v1\.2\.0 and later versions of the SageMaker Edge Manager component\.  
\(Optional\) The amount of time \(in minutes\) between which this component checks for new models to download\. This option applies when you set `DeploymentEnable` to `true`\.  
Default: `1440` \(1 day\)

`DLRBackendOptions`  
This feature is available in v1\.2\.0 and later versions of the SageMaker Edge Manager component\.  
\(Optional\) The DLR runtime flags to set in the DLR runtime that this component uses\. You can set the following flag:  
+ `TVM_TENSORRT_CACHE_DIR` – Enable TensorRT model caching\. Specify an absolute path to an existing folder that has read/write permissions\.
+ `TVM_TENSORRT_CACHE_DISK_SIZE_MB` – Assigns the upper limit of the TensorRT model cache folder\. When the directory size grows beyond this limit the cached engines that are used the least are deleted\. The default value is 512 MB\.
For example, you can set this parameter to the following value to enable TensorRT model caching and limit the cache size to 800 MB\.  

```
TVM_TENSORRT_CACHE_DIR=/data/secured_folder/trt/cache; TVM_TENSORRT_CACHE_DISK_SIZE_MB=800
```

`SagemakerEdgeLogVerbose`  
\(Optional\) String value that specifies whether to enable debug logging\. Supported values are `true` and `false`\.  
Default: `false`

`UnixSocketName`  
\(Optional\) The location of the SageMaker Edge Manager socket file descriptor on the core device\.  
Default: `/tmp/aws.greengrass.SageMakerEdgeManager.sock`

**Example: Configuration merge update**  
The following example configuration specifies that the core device is part of the *MyEdgeDeviceFleet* and that the agent writes capture data both to the device and to an S3 bucket\. This configuration also enables debug logging\.  

```
{
    "DeviceFleetName": "MyEdgeDeviceFleet",
    "BucketName": "DOC-EXAMPLE-BUCKET",
    "CaptureDataDestination": "Disk",
    "CaptureDataPeriodicUpload": "true",
    "SagemakerEdgeLogVerbose": "true"    
}
```

## Local log file<a name="sagemaker-edge-manager-component-log-file"></a>

This component uses the following log file\.

------
#### [ Linux ]

```
/greengrass/v2/logs/aws.greengrass.SageMakerEdgeManager.log
```

------
#### [ Windows ]

```
C:\greengrass\v2\logs\aws.greengrass.SageMakerEdgeManager.log
```

------

**To view this component's logs**
+ Run the following command on the core device to view this component's log file in real time\. Replace */greengrass/v2* or *C:\\greengrass\\v2* with the path to the AWS IoT Greengrass root folder\.

------
#### [ Linux ]

  ```
  sudo tail -f /greengrass/v2/logs/aws.greengrass.SageMakerEdgeManager.log
  ```

------
#### [ Windows \(PowerShell\) ]

  ```
  Get-Content C:\greengrass\v2\logs\aws.greengrass.SageMakerEdgeManager.log -Tail 10 -Wait
  ```

------

## Changelog<a name="sagemaker-edge-manager-component-changelog"></a>

The following table describes the changes in each version of the component\.


|  **Version**  |  **Changes**  | 
| --- | --- | 
|  1\.3\.1  | Version updated for Greengrass nucleus version 2\.8\.0 release\. | 
|  1\.3\.0  |  [\[See the AWS documentation website for more details\]](http://docs.aws.amazon.com/greengrass/v2/developerguide/sagemaker-edge-manager-component.html)  | 
|  1\.2\.0  |  [\[See the AWS documentation website for more details\]](http://docs.aws.amazon.com/greengrass/v2/developerguide/sagemaker-edge-manager-component.html)  | 
|  1\.1\.1  |  Version updated for Greengrass nucleus version 2\.7\.0 release\.  | 
|  1\.1\.0  |  [\[See the AWS documentation website for more details\]](http://docs.aws.amazon.com/greengrass/v2/developerguide/sagemaker-edge-manager-component.html)  | 
|  1\.0\.3  |  Version updated for Greengrass nucleus version 2\.4\.0 release\.  | 
|  1\.0\.2  |  [\[See the AWS documentation website for more details\]](http://docs.aws.amazon.com/greengrass/v2/developerguide/sagemaker-edge-manager-component.html)  | 
|  1\.0\.1  |  Version updated for Greengrass nucleus version 2\.3\.0 release\.  | 
|  1\.0\.0  |  Initial version\.  | 