FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose dev server port
EXPOSE 8080

# Start Vite dev server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
