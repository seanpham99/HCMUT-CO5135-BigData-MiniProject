COMPOSE_DATA := docker compose -f docker-compose.yml
AIRFLOW_SERVICES := airflow-init airflow-apiserver airflow-scheduler airflow-dag-processor airflow-worker airflow-triggerer flower

.PHONY: airflow-refresh airflow-data-up validate

# Rebuild and recreate Airflow services after DAG or dependency changes.
airflow-refresh:
	$(COMPOSE_DATA) up -d --build --force-recreate $(AIRFLOW_SERVICES)

# Start the full data stack without forced recreation.
airflow-data-up:
	$(COMPOSE_DATA) up -d

validate:
	./scripts/validate.sh
