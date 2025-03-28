from pipeline import get_pipeline
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

role = "arn:aws:iam::915992498469:role/service-role/AmazonSageMaker-ExecutionRole-20250326T110603"
pipeline_session = PipelineSession()

# build the pipeline
pipeline = get_pipeline(pipeline_session=pipeline_session, role=role)

pipeline.upsert(role_arn=role)
print(f"✅ Pipeline '{pipeline.name}' created or updated.")

execution = pipeline.start()
print(f"🚀 Execution started. Execution ARN: {execution.arn}")

execution.wait()
print("🎉 Pipeline execution completed!")
