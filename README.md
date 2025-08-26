# Recipe Bot

Monorepo split into backend and frontend with CI that builds Docker images and pushes to ECR.

Repository layout
- backend/ — Python Lambda-compatible backend
  - `backend/app_pipeline.py` — pipeline used by Lambda and local scripts
  - `backend/lambda_handler.py` — Lambda webhook entrypoint
  - `backend/Dockerfile` — container image for Lambda (uses awslambdaric)
  - `backend/requirements.txt`
- frontend/ — static frontend placeholder
  - `frontend/index.html`
  - `frontend/Dockerfile` — serves static site with Nginx
- smoke_test.py — basic smoke test (imports `backend.lambda_handler`)
- .github/workflows/ci-cd.yml — CI workflow: lint/test/build/push to ECR

CI / CD
- The GitHub Actions workflow builds backend and frontend images and pushes them to ECR:
  - ECR registry: `ACC_ID.dkr.ecr.REGION.amazonaws.com`
  - Images pushed:
    - `.../recipe-bot:backend-<sha>` and `.../recipe-bot:backend-latest`
    - `.../recipe-bot:frontend-<sha>` and `.../recipe-bot:frontend-latest`
- Required GitHub secrets:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`

How to run locally
- Build backend image:
  docker build -t recipe-bot-backend:local -f backend/Dockerfile backend
- Build frontend image:
  docker build -t recipe-bot-frontend:local -f frontend/Dockerfile frontend

Deployment
- The workflow currently stops after pushing images to ECR and prints a placeholder.
- Tell me where you want to deploy (ECS service name, cluster, Terraform, EKS, or other), and I will add the CD step to update the running service.

Notes / Next steps
- Consider storing IG cookies in S3 and setting `IG_COOKIES_S3` env var for the backend.
- To enable smoke tests in CI, uncomment the step in `.github/workflows/ci-cd.yml`.
