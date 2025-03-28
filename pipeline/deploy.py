# deploy.py
from pipeline import pipeline  # Make sure pipeline.py is in the same directory
from sagemaker.workflow.pipeline import Pipeline

# Create or update pipeline
pipeline.upsert(role_arn=role)
print(f"✅ Pipeline '{pipeline.name}' created or updated.")

# Start pipeline execution
execution = pipeline.start()
print(f"🚀 Execution started. Execution ARN: {execution.arn}")

# Wait for execution to complete (optional)
execution.wait()
print("🎉 Pipeline execution completed!")
