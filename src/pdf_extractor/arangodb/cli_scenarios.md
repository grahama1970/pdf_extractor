# Message Search Commands Usage Scenarios

## Basic Message Search

| Field | Description |
|-------|-------------|
| **Command** | `search messages "configuration"` |
| **Question** | How do I find messages about configuration settings? |
| **Usage** | When I need to retrieve past conversations about system configuration |
| **Results** | Returns a list of messages containing configuration information |
| **Outcome** | I can quickly find relevant messages about configuration settings |

## Message Search with Type Filter

| Field | Description |
|-------|-------------|
| **Command** | `search messages "user question" --message-type USER` |
| **Question** | What questions have users been asking? |
| **Usage** | When I need to analyze common user queries |
| **Results** | Returns recent user messages across all conversations |
| **Outcome** | I can identify patterns in user questions and needs |

## Message Search with Conversation Filter

| Field | Description |
|-------|-------------|
| **Command** | `search messages "" --conversation 12345-abcde-67890` |
| **Question** | What was discussed in a specific conversation? |
| **Usage** | When I need to review all messages from a particular conversation |
| **Results** | Returns all messages from the specified conversation |
| **Outcome** | I can see the entire conversation history in chronological order |

## Unified Search Across Collections

| Field | Description |
|-------|-------------|
| **Command** | `search unified "error handling"` |
| **Question** | Where can I find information about error handling across both documents and conversations? |
| **Usage** | When I need comprehensive information that could be in either documentation or chat history |
| **Results** | Returns a combined list of documents and messages related to error handling |
| **Outcome** | I get a complete view of error handling information without having to do separate searches |

## Search Only Documents

| Field | Description |
|-------|-------------|
| **Command** | `search unified "api reference" --collections documents` |
| **Question** | How do I search only in documents? |
| **Usage** | When I need to find information in technical documentation without message results |
| **Results** | Returns only document results matching the query |
| **Outcome** | I can quickly find relevant documentation without message noise |

## Exclude Messages from Search

| Field | Description |
|-------|-------------|
| **Command** | `search unified "security protocol" --exclude messages` |
| **Question** | How do I exclude messages from my search? |
| **Usage** | When I want to search everywhere except message history |
| **Results** | Returns results from all collections except messages |
| **Outcome** | I get focused results from documentation and other data sources |

## Semantic Search

| Field | Description |
|-------|-------------|
| **Command** | `search unified "improving system performance" --type semantic` |
| **Question** | How do I find conceptually similar content rather than exact keyword matches? |
| **Usage** | When I need to find content that's semantically related to my query |
| **Results** | Returns semantically related content even if it doesn't contain the exact keywords |
| **Outcome** | I discover relevant information that keyword search might miss |

## JSON Output

| Field | Description |
|-------|-------------|
| **Command** | `search messages "database error" --json` |
| **Question** | How can I get search results in a machine-readable format? |
| **Usage** | When I need to process search results programmatically |
| **Results** | Returns search results in JSON format |
| **Outcome** | I can pipe the results to other tools or save them for later analysis |