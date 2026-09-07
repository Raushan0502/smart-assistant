-- Smart Inbox Assistant -- audit and statistics package.
--
-- The assignment specifies Oracle (PL/SQL). Tables come from Django
-- migrations so the ORM and migration history stay consistent, but the logic
-- below is deliberately in the database rather than in Python, because it
-- belongs there:
--
--   * SI_AUDIT.LOG_EVENT is called from a trigger, so an audit row is written
--     even if a future code path forgets to. An audit trail that depends on
--     every caller remembering is not an audit trail.
--
--   * The reporting functions aggregate across thousands of rows. Doing that
--     in the database avoids pulling the rows into the application to count
--     them, which is the usual reason a queue view gets slow.
--
-- Apply with:  python db/apply_plsql.py

-- Sequence used by LOG_EVENT. Django manages its own identity columns, so the
-- trigger path needs an explicit sequence to insert with.
DECLARE
    v_exists NUMBER;
BEGIN
    SELECT COUNT(*) INTO v_exists
      FROM user_sequences WHERE sequence_name = 'INBOX_AUDITEVENT_SEQ';
    IF v_exists = 0 THEN
        EXECUTE IMMEDIATE
            'CREATE SEQUENCE inbox_auditevent_seq START WITH 100000 INCREMENT BY 1';
    END IF;
END;
/

CREATE OR REPLACE PACKAGE si_audit AS

    -- Record one AI call or system event against a message.
    PROCEDURE log_event(
        p_message_id   IN NUMBER,
        p_event_type   IN VARCHAR2,
        p_model_name   IN VARCHAR2 DEFAULT NULL,
        p_succeeded    IN NUMBER   DEFAULT 1,
        p_latency_ms   IN NUMBER   DEFAULT 0,
        p_error        IN VARCHAR2 DEFAULT NULL
    );

    -- Proportion of extracted fields the source actually stated, 0..1.
    -- A low value means thin source documents, not failed extraction.
    FUNCTION completeness_for_message(p_message_id IN NUMBER) RETURN NUMBER;

    -- Count of fields whose supporting quote could not be verified against the
    -- source. Anything above zero needs human attention.
    FUNCTION unverified_field_count(p_message_id IN NUMBER) RETURN NUMBER;

    -- Mean processing time in milliseconds across messages that completed.
    FUNCTION mean_processing_ms RETURN NUMBER;

    -- Whether a message has been touched by a human reviewer.
    FUNCTION has_been_reviewed(p_message_id IN NUMBER) RETURN NUMBER;

END si_audit;
/

CREATE OR REPLACE PACKAGE BODY si_audit AS

    PROCEDURE log_event(
        p_message_id   IN NUMBER,
        p_event_type   IN VARCHAR2,
        p_model_name   IN VARCHAR2 DEFAULT NULL,
        p_succeeded    IN NUMBER   DEFAULT 1,
        p_latency_ms   IN NUMBER   DEFAULT 0,
        p_error        IN VARCHAR2 DEFAULT NULL
    ) IS
        -- Autonomous so an audit row survives a rollback of the work it
        -- describes. If a message fails and its transaction is rolled back,
        -- the record that it was attempted must still be there.
        PRAGMA AUTONOMOUS_TRANSACTION;
    BEGIN
        INSERT INTO inbox_auditevent (
            id, message_id, event_type, model_name, is_stub,
            prompt_sha256, prompt_chars, latency_ms,
            succeeded, error, response_summary, created_at
        ) VALUES (
            inbox_auditevent_seq.NEXTVAL, p_message_id, p_event_type,
            NVL(p_model_name, 'unknown'), 0,
            '', 0, NVL(p_latency_ms, 0),
            NVL(p_succeeded, 1), NVL(p_error, ''), '{}', SYSTIMESTAMP
        );
        COMMIT;
    EXCEPTION
        WHEN OTHERS THEN
            -- Auditing must never break the operation it is recording.
            ROLLBACK;
    END log_event;

    FUNCTION completeness_for_message(p_message_id IN NUMBER) RETURN NUMBER IS
        v_total  NUMBER := 0;
        v_stated NUMBER := 0;
    BEGIN
        SELECT COUNT(*),
               COUNT(CASE
                         WHEN LOWER(TRIM(f.value)) NOT IN
                              ('not stated', 'unknown', 'n/a', '')
                         THEN 1
                     END)
          INTO v_total, v_stated
          FROM inbox_extractedfield f
          JOIN inbox_extraction e ON e.id = f.extraction_id
         WHERE e.message_id = p_message_id;

        IF v_total = 0 THEN
            RETURN 0;
        END IF;
        RETURN ROUND(v_stated / v_total, 3);
    END completeness_for_message;

    FUNCTION unverified_field_count(p_message_id IN NUMBER) RETURN NUMBER IS
        v_count NUMBER := 0;
    BEGIN
        SELECT COUNT(*)
          INTO v_count
          FROM inbox_extractedfield f
          JOIN inbox_extraction e ON e.id = f.extraction_id
         WHERE e.message_id = p_message_id
           AND f.quote_verified = 0
           AND LOWER(TRIM(f.value)) NOT IN ('not stated', 'unknown', 'n/a', '');
        RETURN v_count;
    END unverified_field_count;

    FUNCTION mean_processing_ms RETURN NUMBER IS
        v_mean NUMBER;
    BEGIN
        SELECT AVG(processing_ms)
          INTO v_mean
          FROM inbox_message
         WHERE processing_ms IS NOT NULL;
        RETURN NVL(ROUND(v_mean, 1), 0);
    END mean_processing_ms;

    FUNCTION has_been_reviewed(p_message_id IN NUMBER) RETURN NUMBER IS
        v_count NUMBER := 0;
    BEGIN
        SELECT COUNT(*)
          INTO v_count
          FROM inbox_reviewaction
         WHERE message_id = p_message_id;
        RETURN CASE WHEN v_count > 0 THEN 1 ELSE 0 END;
    END has_been_reviewed;

END si_audit;
/

-- Guarantee a review action is always timestamped, whatever inserts it.
-- The assignment requires every reviewer action to carry a timestamp; a
-- database default cannot be bypassed by an application that forgets.
CREATE OR REPLACE TRIGGER si_review_action_stamp
    BEFORE INSERT ON inbox_reviewaction
    FOR EACH ROW
BEGIN
    IF :NEW.created_at IS NULL THEN
        :NEW.created_at := SYSTIMESTAMP;
    END IF;
END;
/

-- Queue view: what the reviewer screen lists, assembled in the database so the
-- API does not issue a query per message to build the same picture.
CREATE OR REPLACE VIEW si_review_queue AS
SELECT
    m.id                                    AS message_id,
    m.message_id                            AS external_id,
    m.subject,
    m.sender,
    m.sent_at,
    m.received_at,
    m.processing_status,
    m.review_status,
    m.processing_ms,
    si_audit.completeness_for_message(m.id) AS completeness,
    si_audit.unverified_field_count(m.id)   AS unverified_fields,
    si_audit.has_been_reviewed(m.id)        AS reviewed,
    (SELECT COUNT(*) FROM inbox_document d
      WHERE d.message_id = m.id)            AS document_count,
    (SELECT LISTAGG(c.category, ',') WITHIN GROUP (ORDER BY c.confidence DESC)
       FROM inbox_classification c
      WHERE c.message_id = m.id AND c.applies = 1) AS categories,
    (SELECT MAX(c.confidence) FROM inbox_classification c
      WHERE c.message_id = m.id AND c.applies = 1) AS top_confidence
FROM inbox_message m;
