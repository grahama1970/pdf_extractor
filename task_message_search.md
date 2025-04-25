# Task Plan: Implementing Search Capabilities for Message History

## Overview
This plan outlines the tasks needed to integrate message history into the search system by adding embeddings, view configurations, and search functions that work across collections.

## Tasks

### 1. Add Embedding Field to Message History
- [ ] Modify the message creation process to include embeddings
- [ ] Update all relevant functions that create messages to call the embedding utility
- [ ] Validate that messages include embeddings for semantic search

### 2. Update Search View Configuration
- [ ] Modify the view configuration to include the message history collection
- [ ] Add appropriate field mappings for searchable fields in messages
- [ ] Configure analyzers for content and metadata fields
- [ ] Ensure vector indices for message embeddings

### 3. Create Collection-Aware Search Functions
- [ ] Implement unified search functions that work across collections
- [ ] Add collection filtering to search across both messages and documents
- [ ] Create collection exclusion functionality
- [ ] Ensure backward compatibility with existing search functions

### 4. Add CLI Commands for Message Search
- [ ] Create message-specific search commands
- [ ] Implement unified search with collection filtering options
- [ ] Add rich formatting for search results including collection origin
- [ ] Update documentation with search type guidance for agents

### 5. Test and Validate Changes
- [ ] Create test messages with embeddings
- [ ] Test message-only searches using different search types
- [ ] Test cross-collection searches with different filtering options
- [ ] Validate that existing functionality is preserved
