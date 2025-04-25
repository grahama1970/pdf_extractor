FROM python:3.11-slim
WORKDIR /app

# Install LiteLLM
RUN pip install --upgrade pip
RUN pip install "litellm[proxy]"

# Copy config into the container
COPY litellm_config.yaml /config/litellm_config.yaml

# Expose the proxy port
EXPOSE 4000

# Start the LiteLLM proxy
CMD ["litellm", "--config", "/config/litellm_config.yaml"]
