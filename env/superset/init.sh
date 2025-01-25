#!/bin/bash
set -e

# Wait for dependencies
until superset db upgrade; do
    echo "Upgrading DB..."
    sleep 3
done

# Create admin user
superset fab create-admin \
    --username $ADMIN_USERNAME \
    --firstname $ADMIN_FIRSTNAME \
    --lastname $ADMIN_LASTNAME \
    --email $ADMIN_EMAIL \
    --password $ADMIN_PASSWORD

# Initialize roles and permissions
superset init

# Load examples if requested
if [ "$SUPERSET_LOAD_EXAMPLES" = "yes" ]; then
    superset load_examples
fi

# Start Superset
/usr/bin/run-server.sh