/**
 * The single place the UI talks to the backend.
 *
 * Components never construct URLs or know about pagination shapes; keeping
 * that here means an API change is one file to update, and it keeps the
 * components readable as UI rather than as HTTP plumbing.
 */
import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import {
  MessageDetail,
  Paginated,
  QueueRow,
  ScreeningResult,
  SystemStatus,
} from './models';

const API_BASE = 'http://localhost:8080/api';

export interface QueueFilters {
  category?: string;
  review_status?: string;
  processing_status?: string;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);

  /** The review queue, optionally filtered. */
  queue(filters: QueueFilters = {}): Observable<Paginated<QueueRow>> {
    let params = new HttpParams();
    for (const [key, value] of Object.entries(filters)) {
      if (value) {
        params = params.set(key, value);
      }
    }
    return this.http.get<Paginated<QueueRow>>(`${API_BASE}/messages/`, { params });
  }

  /** One message with everything: documents, fields, provenance, audit. */
  message(id: number): Observable<MessageDetail> {
    return this.http.get<MessageDetail>(`${API_BASE}/messages/${id}/`);
  }

  /** Accept the AI's output as it stands. */
  accept(id: number, reviewer: string, note = ''): Observable<MessageDetail> {
    return this.http.post<MessageDetail>(`${API_BASE}/messages/${id}/accept/`, {
      reviewer,
      note,
    });
  }

  /** Override one extracted field. The original is preserved server-side. */
  overrideField(
    id: number,
    fieldName: string,
    newValue: string,
    reviewer: string,
  ): Observable<MessageDetail> {
    return this.http.post<MessageDetail>(
      `${API_BASE}/messages/${id}/override_field/`,
      { field_name: fieldName, new_value: newValue, reviewer },
    );
  }

  /** Add or remove a category the AI got wrong. */
  overrideCategory(
    id: number,
    category: string,
    applies: boolean,
    reviewer: string,
    note = '',
  ): Observable<MessageDetail> {
    return this.http.post<MessageDetail>(
      `${API_BASE}/messages/${id}/override_category/`,
      { category, applies, reviewer, note },
    );
  }

  /** Poll the mailbox. Returns as soon as the work is queued. */
  ingest(): Observable<{ queued: number }> {
    return this.http.post<{ queued: number }>(`${API_BASE}/ingest/`, {});
  }

  /** Queue depth, AI service health, message counts. */
  status(): Observable<SystemStatus> {
    return this.http.get<SystemStatus>(`${API_BASE}/status/`);
  }

  /** Literature screening for an independently uploaded article. */
  screenArticle(file: File): Observable<ScreeningResult> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<ScreeningResult>(`${API_BASE}/screen-article/`, form);
  }
}
