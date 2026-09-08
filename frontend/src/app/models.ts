/**
 * Types mirroring the backend API.
 *
 * Kept as interfaces rather than classes: these are wire shapes, not domain
 * objects with behaviour, and the compiler is what stops the UI drifting from
 * what the API actually returns.
 */

export type Category = 'ICSR' | 'PQC' | 'MI' | 'NOT_RELEVANT';

export interface QueueRow {
  id: number;
  message_id: string;
  subject: string;
  sender: string;
  sent_at: string | null;
  received_at: string;
  processing_status: string;
  review_status: string;
  processing_ms: number | null;
  categories: Category[];
  top_confidence: number | null;
  document_count: number;
  /** True when the AI's output should not be taken at face value. */
  needs_attention: boolean;
}

export interface Paginated<T> {
  count: number;
  next: string | null;
  previous: string | null;
  results: T[];
}

export interface Verdict {
  category: Category;
  applies: boolean;
  confidence: number;
  reason: string;
  model_name: string;
}

export interface Field {
  name: string;
  value: string;
  confidence: number;
  is_stated: boolean;
  source_document: string;
  source_page: number | null;
  source_quote: string;
  /** False when the supporting quote could not be found in the source. */
  quote_verified: boolean;
}

export interface Extraction {
  category: Category;
  narrative: string;
  completeness: number;
  model_name: string;
  fields_data: Field[];
}

export interface TableBlock {
  header: string[];
  rows: string[][];
  page: number | null;
}

export interface ImageBlock {
  width: number;
  height: number;
  description: string | null;
  needs_review: boolean;
  page: number | null;
}

export interface DocumentRecord {
  document_id: string;
  file_name: string;
  media_type: string;
  flavour: string;
  language: string;
  page_count: number;
  processed: boolean;
  extracted_text: string;
  tables: TableBlock[];
  images: ImageBlock[];
  doc_metadata: Record<string, string>;
  warnings: string[];
  summary: string;
  looks_relevant: boolean | null;
  relevance_reason: string;
  summary_confidence: number | null;
  ocr_confidence: number | null;
}

export interface ReviewActionRecord {
  id: number;
  action: string;
  reviewer: string;
  created_at: string;
  field_name: string;
  original_value: string;
  new_value: string;
  category: string;
  note: string;
}

export interface AuditEventRecord {
  event_type: string;
  model_name: string;
  is_stub: boolean;
  prompt_sha256: string;
  prompt_chars: number;
  latency_ms: number;
  succeeded: boolean;
  error: string;
  created_at: string;
}

export interface MessageDetail {
  id: number;
  message_id: string;
  subject: string;
  sender: string;
  recipient: string;
  sent_at: string | null;
  received_at: string;
  body_text: string;
  language: string;
  processing_status: string;
  review_status: string;
  processing_ms: number | null;
  error: string;
  warnings: string[];
  headers: Record<string, string>;
  categories: Category[];
  documents: DocumentRecord[];
  classifications: Verdict[];
  extractions: Extraction[];
  review_actions: ReviewActionRecord[];
  audit_events: AuditEventRecord[];
}

export interface SystemStatus {
  queue: {
    pending: number;
    processed: number;
    failed: number;
    workers: number;
    running: boolean;
  };
  ai_service: {
    status: string;
    model?: string;
    is_stub?: boolean;
    note?: string;
    detail?: string;
  };
  messages: { total: number; pending_review: number; failed: number };
}

export interface ScreeningResult {
  document: DocumentRecord;
  screening: {
    document_id: string;
    summary: string;
    looks_relevant: boolean;
    relevance_reason: string;
    confidence: number;
    model: string;
  };
}

export const CATEGORY_LABELS: Record<Category, string> = {
  ICSR: 'Safety Report',
  PQC: 'Quality Complaint',
  MI: 'Info Request',
  NOT_RELEVANT: 'Not Relevant',
};
