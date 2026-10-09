FROM public.ecr.aws/docker/library/python:3.13-slim

RUN pip install uv

WORKDIR /app

# Copy everything first
COPY . .

# Install dependencies
RUN uv sync --frozen

RUN uv run python -m churn_mlops.train

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "churn_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]