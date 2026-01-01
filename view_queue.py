# %%
import json
import base64
from azure.identity import DefaultAzureCredential
from azure.storage.queue import QueueClient
from azure.storage.filedatalake import DataLakeServiceClient

# From your setup
ACCOUNT = "cyrtdata"                       # storage account name
QUEUE   = "forecast-data"                  # queue

cred = DefaultAzureCredential()
queue_url = f"https://{ACCOUNT}.queue.core.windows.net"

qc = QueueClient(account_url=queue_url, queue_name=QUEUE, credential=cred)

# Example: receive one message
msgs = qc.receive_messages(messages_per_page=32, visibility_timeout=600)
print(msgs)
for m in msgs:
    print(m)
    # Azure Storage Queue messages are base64-encoded by default
    try:
        m.pop_receipt
        decoded_content = base64.b64decode(m.content).decode('utf-8')
        message = json.loads(decoded_content)
        print(message)
    except Exception as e:
        print(f"Failed to decode message: {e}")
        print(f"Raw message content: {repr(m.content)}")
        continue
    
    api_type = message.get("data", {}).get("api")
    if api_type == "FlushWithClose":
        file_url = message["data"]["url"]
        print(f"File location: {file_url}")
    else:
        print(f"Skipping message with unsupported API type: {api_type}")

# %%
properties = qc.get_queue_properties()
print(f"message count: {properties.approximate_message_count}")


# %%
