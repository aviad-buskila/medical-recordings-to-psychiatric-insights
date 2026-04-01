CREATE TABLE IF NOT EXISTS clinical_ai.documents (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clinical_ai.document_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    source_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    embedding VECTOR(768),
    bm25_tokens TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_source_id
    ON clinical_ai.document_chunks(source_id);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON clinical_ai.document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
