#!/usr/bin/env bash

# Load PostgreSQL environment variables from GCP Secrets
# Usage: source resources/load_postgres_env.sh

GCP_PROJECT="873471276793"

echo "🔐 Loading PostgreSQL environment variables from project '$GCP_PROJECT'..."

# List of PostgreSQL environment variables to load
PG_VARS=("PGHOST" "PGPORT" "PGDATABASE" "PGUSER" "PGPASSWORD")

for var in "${PG_VARS[@]}"; do
    echo "📥 Loading $var..."
    SECRET_VALUE=$(gcloud secrets versions access latest \
      --project "$GCP_PROJECT" \
      --secret "$var" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$SECRET_VALUE" ]; then
        export "$var"="$SECRET_VALUE"
        echo "✅ Loaded $var"
    else
        echo "❌ Failed to load $var"
    fi
done

echo "🎉 Finished loading PostgreSQL environment variables"
echo ""
echo "Current PostgreSQL environment variables:"
for var in "${PG_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        if [ "$var" == "PGPASSWORD" ]; then
            echo "  $var=***hidden***"
        else
            echo "  $var=${!var}"
        fi
    fi
done