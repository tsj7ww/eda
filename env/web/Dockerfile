FROM node:18-bullseye-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files first for better caching
COPY workspace/web/package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY workspace/web .

EXPOSE 3000

# Start the server
CMD ["npm", "run", "dev"]
# CMD ["node", "server.js"]