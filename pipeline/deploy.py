# deploy.py
from pipeline import pipeline  # Make sure pipeline.py is in the same directory
from sagemaker.workflow.pipeline import Pipeline

# Create or update pipeline
pipeline_session = PipelineSession()
pipeline.upsert(role_arn="arn:aws:iam::915992498469:role/service-role/AmazonSageMaker-ExecutionRole-20250326T110603")
print(f"✅ Pipeline '{pipeline.name}' created or updated.")

# Start pipeline execution
execution = pipeline.start()
print(f"🚀 Execution started. Execution ARN: {execution.arn}")

# Wait for execution to complete (optional)
execution.wait()
print("🎉 Pipeline execution completed!")
