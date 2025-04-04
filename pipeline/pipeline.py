# pipeline.py
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model

role = "arn:aws:iam::915992498469:role/service-role/AmazonSageMaker-ExecutionRole-20250326T110603"
pipeline_session = PipelineSession()
bucket = 'cifake-mlops'

# Parameters
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
batch_size = ParameterInteger(name="BatchSize", default_value=64)
epochs = ParameterInteger(name="Epochs", default_value=10)
learning_rate = ParameterFloat(name="LearningRate", default_value=0.0001)
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")

# Step 1: Preprocessing
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type=processing_instance_type,
    instance_count=1,
    role=role,
    base_job_name="cifake-preprocess",
    sagemaker_session=pipeline_session
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=f"s3://{bucket}/CIFAKE_Dataset", destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(output_name="processed_data", source="/opt/ml/processing/output")
    ],
    code='src/preprocess.py'
)

# Step 2: Training
estimator = PyTorch(
    entry_point='src/train.py',
    role=role,
    instance_count=1,
    instance_type=training_instance_type,
    framework_version="2.1.0",
    py_version="py310",
    output_path=f"s3://{bucket}/model-artifacts",
    sagemaker_session=pipeline_session,
    hyperparameters={
        "epochs": epochs,
        "batch-size": batch_size,
        "lr": learning_rate
    }
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={"training": processing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri}
)

# Step 3: Register Model (✅ use PyTorchModel instead of generic Model)
model = Model(
    image_uri=sagemaker.image_uris.retrieve(
        framework='pytorch',
        region=pipeline_session.boto_region_name,
        version='2.1.0',
        py_version='py310',
        instance_type='ml.m5.xlarge',
        image_scope='inference'
    ),
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    entry_point='src/inference.py',
    role=role,
    sagemaker_session=pipeline_session
)

register_step = ModelStep(
    name="RegisterTrainedModel",
    step_args=model.create(
        instance_type="ml.m5.xlarge"
    )
)

def get_pipeline(pipeline_session, role):
    return Pipeline(
        name="CIFakeDetectionPipeline",
        parameters=[
            processing_instance_type,
            training_instance_type,
            batch_size,
            epochs,
            learning_rate,
            model_approval_status
        ],
        steps=[processing_step, training_step, register_step],
        sagemaker_session=pipeline_session
    )

