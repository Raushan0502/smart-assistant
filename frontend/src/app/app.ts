/**
 * The reviewer screen.
 *
 * One component holding a queue and a detail panel. That is a deliberate
 * choice for a prototype of this size: splitting it across five components
 * plus a store would add indirection without adding capability, and the brief
 * asks for a *simple* review screen. The API layer is separated because that
 * boundary earns its keep; this one would not.
 *
 * The screen is built around one idea: a reviewer should be able to answer
 * "why does the AI think that, and where did it come from?" without leaving
 * the page. So every field shows its confidence and its source quote, and
 * every category shows the reason the model gave.
 */
import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ApiService, QueueFilters } from './api.service';
import {
  CATEGORY_LABELS,
  Category,
  DocumentRecord,
  Field,
  MessageDetail,
  QueueRow,
  ScreeningResult,
  SystemStatus,
} from './models';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit, OnDestroy {
  private api = inject(ApiService);

  readonly categoryLabels = CATEGORY_LABELS;
  readonly allCategories: Category[] = ['ICSR', 'PQC', 'MI', 'NOT_RELEVANT'];

  rows = signal<QueueRow[]>([]);
  selected = signal<MessageDetail | null>(null);
  status = signal<SystemStatus | null>(null);
  loading = signal(false);
  busy = signal(false);
  error = signal('');

  filters: QueueFilters = {};
  reviewer = 'demo-reviewer';
  activeTab: 'fields' | 'documents' | 'audit' = 'fields';

  /** Local edits, keyed by field name, so the input is not bound to the model. */
  edits: Record<string, string> = {};

  /** Set when a call failed because every provider is out of allowance. */
  quotaMessage = signal('');

  screening = signal<ScreeningResult | null>(null);
  screeningBusy = signal(false);

  private statusTimer?: number;

  ngOnInit(): void {
    this.refresh();
    this.pollStatus();
    // The queue changes while work drains, so status is polled rather than
    // fetched once. Ten seconds is frequent enough to feel live without
    // hammering the API.
    this.statusTimer = window.setInterval(() => this.pollStatus(), 10_000);
  }

  ngOnDestroy(): void {
    if (this.statusTimer) {
      window.clearInterval(this.statusTimer);
    }
  }

  pollStatus(): void {
    this.api.status().subscribe({
      next: (s) => this.status.set(s),
      error: () => this.status.set(null),
    });
  }

  refresh(): void {
    this.loading.set(true);
    this.error.set('');
    this.api.queue(this.filters).subscribe({
      next: (page) => {
        this.rows.set(page.results);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(
          `Could not load the queue: ${err.message ?? err}. Is the backend running on :8080?`,
        );
        this.loading.set(false);
      },
    });
  }

  open(row: QueueRow): void {
    this.api.message(row.id).subscribe({
      next: (detail) => {
        this.selected.set(detail);
        this.edits = {};
        this.activeTab = 'fields';
      },
      error: (err) => this.error.set(`Could not open message: ${err.message ?? err}`),
    });
  }

  ingest(): void {
    this.busy.set(true);
    this.api.ingest().subscribe({
      next: () => {
        this.busy.set(false);
        // Give the workers a moment before re-reading, since processing is
        // asynchronous by design.
        window.setTimeout(() => this.refresh(), 1500);
      },
      error: (err) => {
        this.busy.set(false);
        this.report(err, 'Ingest failed');
      },
    });
  }

  accept(): void {
    const message = this.selected();
    if (!message) return;
    this.busy.set(true);
    this.api.accept(message.id, this.reviewer).subscribe({
      next: (updated) => {
        this.selected.set(updated);
        this.busy.set(false);
        this.refresh();
      },
      error: () => this.busy.set(false),
    });
  }

  saveField(field: Field): void {
    const message = this.selected();
    const value = this.edits[field.name];
    if (!message || value === undefined || value === field.value) return;

    this.busy.set(true);
    this.api.overrideField(message.id, field.name, value, this.reviewer).subscribe({
      next: (updated) => {
        this.selected.set(updated);
        delete this.edits[field.name];
        this.busy.set(false);
        this.refresh();
      },
      error: () => this.busy.set(false),
    });
  }

  toggleCategory(category: Category, applies: boolean): void {
    const message = this.selected();
    if (!message) return;
    this.busy.set(true);
    this.api
      .overrideCategory(message.id, category, applies, this.reviewer, 'Set from review screen')
      .subscribe({
        next: (updated) => {
          this.selected.set(updated);
          this.busy.set(false);
          this.refresh();
        },
        error: () => this.busy.set(false),
      });
  }

  onArticleSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    this.screeningBusy.set(true);
    this.screening.set(null);
    this.api.screenArticle(file).subscribe({
      next: (result) => {
        this.screening.set(result);
        this.screeningBusy.set(false);
        input.value = '';
      },
      error: (err) => {
        this.screeningBusy.set(false);
        this.report(err, 'Screening failed');
      },
    });
  }

  /** Route an error to the right banner.
   *
   * A spent AI allowance is not a fault: nothing is broken, and the message
   * already says what to do. Showing it as a red error alongside genuine
   * failures would train a reviewer to ignore both.
   */
  private report(err: any, context: string): void {
    const detail = err?.error?.detail ?? err?.message ?? String(err);
    if (err?.status === 429 || /usage limit reached/i.test(detail)) {
      this.quotaMessage.set(detail);
      return;
    }
    this.error.set(`${context}: ${detail}`);
  }

  // ---- display helpers -------------------------------------------------

  /** Confidence bucket, used to colour the badge. */
  confidenceClass(value: number | null): string {
    if (value === null || value === undefined) return 'conf-none';
    if (value >= 0.75) return 'conf-high';
    if (value >= 0.5) return 'conf-mid';
    return 'conf-low';
  }

  percent(value: number | null | undefined): string {
    return value === null || value === undefined ? '--' : `${Math.round(value * 100)}%`;
  }

  fieldLabel(name: string): string {
    return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  }

  editedValue(field: Field): string {
    return this.edits[field.name] ?? field.value;
  }

  isDirty(field: Field): boolean {
    return this.edits[field.name] !== undefined && this.edits[field.name] !== field.value;
  }

  /** Documents that were received but deliberately not parsed. */
  unprocessed(message: MessageDetail): DocumentRecord[] {
    return message.documents.filter((d) => !d.processed);
  }

  verdictFor(message: MessageDetail, category: Category) {
    return message.classifications.find((c) => c.category === category);
  }

  /** True when this message carries anything a reviewer must not miss. */
  hasIntegrityWarning(message: MessageDetail): boolean {
    return message.extractions.some((e) =>
      e.fields_data.some((f) => f.is_stated && !f.quote_verified),
    );
  }
}
