package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/steveyegge/beads/internal/storage"
	"github.com/steveyegge/beads/internal/types"
	"github.com/steveyegge/beads/issueops"
)

// purgeBatchStore covers only the store calls `bd mol wisp gc --closed
// --force` makes. The embedded nil interface makes any other call panic
// loudly rather than return a silent zero value.
type purgeBatchStore struct {
	storage.DoltStorage
	issues  []*types.Issue
	deleter *recordingDeleter
}

func (s *purgeBatchStore) SearchIssues(_ context.Context, _ string, filter types.IssueFilter) ([]*types.Issue, error) {
	if len(filter.IDs) == 0 {
		return s.issues, nil
	}
	var out []*types.Issue
	for _, issue := range s.issues {
		if slices.Contains(filter.IDs, issue.ID) {
			out = append(out, issue)
		}
	}
	return out, nil
}

func (s *purgeBatchStore) GetIssue(_ context.Context, id string) (*types.Issue, error) {
	for _, issue := range s.issues {
		if issue.ID == id {
			return issue, nil
		}
	}
	return nil, storage.ErrNotFound
}

func (s *purgeBatchStore) IsInfraTypeCtx(context.Context, types.IssueType) bool { return false }

func (s *purgeBatchStore) Deleter() (issueops.Deleter, error) { return s.deleter, nil }

// recordingDeleter records each delete request, the unit one transaction
// commits, and fails the failOn-th request (1-based; 0 never fails).
type recordingDeleter struct {
	requests [][]string
	reqs     []issueops.DeleteRequest
	failOn   int
	orphans  func(batch []string) []string
}

func (d *recordingDeleter) Delete(_ context.Context, req issueops.DeleteRequest) (issueops.DeleteResult, error) {
	d.requests = append(d.requests, slices.Clone(req.IDs))
	d.reqs = append(d.reqs, req)
	if len(d.requests) == d.failOn {
		return issueops.DeleteResult{}, errors.New("context deadline exceeded")
	}
	// DeletedIDs as the role reports it: the batch's rows, ascending, once each.
	result := issueops.DeleteResult{DryRun: req.DryRun, Deleted: len(req.IDs), Labels: 2 * len(req.IDs), DeletedIDs: slices.Sorted(slices.Values(req.IDs))}
	if d.orphans != nil {
		result.Orphaned = d.orphans(req.IDs)
	}
	return result, nil
}

// usePurgeBatchStore installs n closed wisps behind the package store for the
// test and returns their ids in candidate order.
func usePurgeBatchStore(t *testing.T, n int, d *recordingDeleter) []string {
	t.Helper()
	ids := make([]string, n)
	issues := make([]*types.Issue, n)
	for i := range n {
		ids[i] = fmt.Sprintf("closed-%04d", i)
		issues[i] = &types.Issue{ID: ids[i], Status: types.StatusClosed, Ephemeral: true, IssueType: types.TypeTask}
	}
	oldStore, oldJSON, oldRootCtx, oldDidWrite := store, jsonOutput, rootCtx, commandDidWrite.Load()
	store = &purgeBatchStore{issues: issues, deleter: d}
	jsonOutput = false
	rootCtx = context.Background()
	commandDidWrite.Store(false)
	commandDeletedIssueIDs.reset()
	t.Cleanup(func() {
		store, jsonOutput, rootCtx = oldStore, oldJSON, oldRootCtx
		commandDidWrite.Store(oldDidWrite)
		commandDeletedIssueIDs.reset()
	})
	return ids
}

// assertExportBookkeeping pins the GH#5896 handoff to auto-export: a write
// happened, and exactly the committed ids are recorded as deleted. Without
// it the export's orphan guard sees them as JSONL-only records and wedges.
func assertExportBookkeeping(t *testing.T, committed, notCommitted []string) {
	t.Helper()
	if !commandDidWrite.Load() {
		t.Error("commandDidWrite is false after committed batches; auto-export would skip the write")
	}
	for _, id := range committed {
		if !commandDeletedIssueIDs.has(id) {
			t.Errorf("committed id %s is not in commandDeletedIssueIDs; the export orphan guard would wedge on it", id)
			break
		}
	}
	for _, id := range notCommitted {
		if commandDeletedIssueIDs.has(id) {
			t.Errorf("id %s of an uncommitted batch is in commandDeletedIssueIDs", id)
			break
		}
	}
}

// TestWispPurgeClosedForceCommitsInBoundedBatches is the regression test for
// GH#4684: `bd mol wisp gc --closed --force` timed out on a large store at
// 120s and again at 300s. The whole candidate set (up to 5000 wisps) went to
// the deleter as ONE request, so one transaction: a run cut short by the
// caller's deadline rolled back to nothing, and every retry started over.
// Each request must stay within the 200 wisps deleteWispBatch commits at once,
// and together the requests must delete every candidate exactly once.
func TestWispPurgeClosedForceCommitsInBoundedBatches(t *testing.T) {
	d := &recordingDeleter{}
	ids := usePurgeBatchStore(t, 401, d)

	captureStdout(t, func() error {
		return runWispPurgeClosed(context.Background(), false, true, nil)
	})

	if len(d.requests) < 2 {
		t.Fatalf("401 closed wisps went out in %d delete request(s); want one per batch of at most 200", len(d.requests))
	}
	var got []string
	for i, batch := range d.requests {
		if len(batch) > 200 {
			t.Errorf("request %d carries %d wisps; one transaction must carry at most 200", i+1, len(batch))
		}
		req := d.reqs[i]
		if !req.Force || req.Cascade || req.DryRun {
			t.Errorf("request %d: Force=%v Cascade=%v DryRun=%v; want a forced, non-cascading, real delete", i+1, req.Force, req.Cascade, req.DryRun)
		}
		got = append(got, batch...)
	}
	if !slices.Equal(got, ids) {
		t.Errorf("batches deleted %d ids; want each of the %d candidates once, in order", len(got), len(ids))
	}
	assertExportBookkeeping(t, ids, nil)
}

// TestWispPurgeClosedForceReportsProgressOnFailure pins what a cut-short run
// says: the batches before the failure are committed, the command stops there,
// and the error tells the caller how far it got and that a rerun continues.
func TestWispPurgeClosedForceReportsProgressOnFailure(t *testing.T) {
	d := &recordingDeleter{failOn: 2}
	ids := usePurgeBatchStore(t, 401, d)

	var err error
	stderr := captureStderr(t, func() {
		captureStdoutIgnoringError(t, func() {
			err = runWispPurgeClosed(context.Background(), false, true, nil)
		})
	})

	if err == nil {
		t.Fatal("a failed batch must fail the command")
	}
	if len(d.requests) != 2 {
		t.Errorf("sent %d request(s) after the second failed; want the run to stop at the failure", len(d.requests))
	}
	for _, want := range []string{"deleted 200 of 401 closed wisp(s)", "run the command again", "context deadline exceeded"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("stderr %q does not say %q", stderr, want)
		}
	}
	assertExportBookkeeping(t, ids[:200], ids[200:])
}

// TestWispPurgeClosedForceOrphansOnlySurvivors: a batch reports as orphaned a
// dependent that a later batch of the same purge deletes. The summary must list
// only the dependents that survive the whole purge, and its deleted_ids
// (GH#6301) must name the rows of every committed batch, not the last one.
func TestWispPurgeClosedForceOrphansOnlySurvivors(t *testing.T) {
	d := &recordingDeleter{orphans: func(batch []string) []string {
		out := []string{"live-step"}
		if !slices.Contains(batch, "closed-0200") {
			out = append(out, "closed-0200")
		}
		return out
	}}
	ids := usePurgeBatchStore(t, 201, d)
	jsonOutput = true

	out := captureStdout(t, func() error {
		return runWispPurgeClosed(context.Background(), false, true, nil)
	})

	var answer struct {
		DeletedCount int      `json:"deleted_count"`
		Labels       int      `json:"labels_removed"`
		Orphaned     []string `json:"orphaned_issues"`
		DeletedIDs   []string `json:"deleted_ids"`
	}
	if err := json.Unmarshal([]byte(out), &answer); err != nil {
		t.Fatalf("want one JSON answer for the whole purge, got %q: %v", out, err)
	}
	if answer.DeletedCount != 201 || answer.Labels != 402 {
		t.Errorf("deleted_count=%d labels_removed=%d; want the batches summed (201, 402)", answer.DeletedCount, answer.Labels)
	}
	if !slices.Equal(answer.Orphaned, []string{"live-step"}) {
		t.Errorf("orphaned_issues = %v; want only the surviving dependent [live-step]", answer.Orphaned)
	}
	if !slices.Equal(answer.DeletedIDs, ids) {
		t.Errorf("deleted_ids names %d row(s); want all %d rows of both batches", len(answer.DeletedIDs), len(ids))
	}
}

// TestDeleteBatchJSONFollowsCallerParameter: deleteBatch prints its result in
// the mode its caller passes, not the package --json flag. `bd mol burn A B
// --force --json` deletes each persistent molecule with deleteBatch(...,
// jsonOutput=false, ...) (mol_burn.go) and prints one BatchBurnResult at the
// end; a per-molecule delete document before it would be read first by a
// consumer decoding one JSON value.
func TestDeleteBatchJSONFollowsCallerParameter(t *testing.T) {
	t.Run("false under global --json prints the text summary", func(t *testing.T) {
		ids := usePurgeBatchStore(t, 2, &recordingDeleter{})
		jsonOutput = true

		out := captureStdout(t, func() error {
			return deleteBatch(nil, ids, true, false, false, false, false, nil, "mol burn")
		})

		if json.Valid([]byte(strings.TrimSpace(out))) {
			t.Fatalf("caller passed jsonOutput=false, got a JSON document: %q", out)
		}
		if !strings.Contains(out, "Deleted 2 issue(s)") {
			t.Errorf("want the text summary, got %q", out)
		}
	})
	t.Run("true without global --json prints the JSON answer", func(t *testing.T) {
		ids := usePurgeBatchStore(t, 2, &recordingDeleter{})

		out := captureStdout(t, func() error {
			return deleteBatch(nil, ids, true, false, false, true, false, nil, "mol burn")
		})

		var answer struct {
			Deleted      []string `json:"deleted"`
			DeletedCount int      `json:"deleted_count"`
		}
		if err := json.Unmarshal([]byte(out), &answer); err != nil {
			t.Fatalf("caller passed jsonOutput=true, want one JSON answer, got %q: %v", out, err)
		}
		if answer.DeletedCount != 2 || !slices.Equal(answer.Deleted, ids) {
			t.Errorf("deleted=%v deleted_count=%d; want %v, 2", answer.Deleted, answer.DeletedCount, ids)
		}
	})
}

// TestDeleteBatchOutputUnchanged: moving deleteBatch's result printing into
// renderDeleteResult changes no byte a delete prints, with or without
// --cascade, in text or --json, and leaves the unconfirmed preview alone.
// Each want is what base prints for the same fixture; the cascaded text
// summary is the only one that names the deleted rows (GH#6301).
func TestDeleteBatchOutputUnchanged(t *testing.T) {
	const text = "✓ Deleted 2 issue(s)\n" +
		"  Removed 0 dependency link(s)\n" +
		"  Removed 4 label(s)\n" +
		"  Removed 0 event(s)\n" +
		"  Updated text references in 0 issue(s)\n"
	const cascadeLine = "  Deleted: closed-0000, closed-0001\n"
	const answer = "{\n" +
		"  \"deleted\": [\n    \"closed-0000\",\n    \"closed-0001\"\n  ],\n" +
		"  \"deleted_count\": 2,\n" +
		"  \"deleted_ids\": [\n    \"closed-0000\",\n    \"closed-0001\"\n  ],\n" +
		"  \"dependencies_removed\": 0,\n" +
		"  \"events_removed\": 0,\n" +
		"  \"labels_removed\": 4,\n" +
		"  \"orphaned_issues\": null,\n" +
		"  \"references_updated\": 0,\n" +
		"  \"schema_version\": 1\n" +
		"}\n"
	const previewHead = "\n⚠️  DELETE PREVIEW\n\n" +
		"Issues to delete (2):\n  closed-0000: \n  closed-0001: \n\n"
	const previewTail = "Would delete: 2 issues\n" +
		"Would remove: 0 dependencies, 4 labels, 0 events\n\n" +
		"This operation cannot be undone!\n"
	const preview = previewHead + previewTail +
		"To proceed, run: bd delete closed-0000 closed-0001 --force\n"
	const previewCascade = previewHead +
		"⚠ Cascade mode enabled - will also delete parent-child descendants; other dependents survive\n\n" +
		previewTail +
		"To proceed with cascade deletion, run: bd delete closed-0000 closed-0001 --cascade --force\n"
	run := func(t *testing.T, force, cascade, asJSON bool) string {
		t.Helper()
		ids := usePurgeBatchStore(t, 2, &recordingDeleter{})
		return captureStdout(t, func() error {
			return deleteBatch(nil, ids, force, false, cascade, asJSON, false, nil, "delete")
		})
	}
	cases := []struct {
		name                   string
		force, cascade, asJSON bool
		want                   string
	}{
		{"text", true, false, false, text},
		{"text cascade", true, true, false, strings.Replace(text, "\n", "\n"+cascadeLine, 1)},
		{"json", true, false, true, answer},
		{"json cascade", true, true, true, answer},
		// The unconfirmed preview never reaches renderDeleteResult.
		{"preview", false, false, false, preview},
		{"preview cascade", false, true, false, previewCascade},
	}
	got := map[string]string{}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got[c.name] = run(t, c.force, c.cascade, c.asJSON)
			if got[c.name] != c.want {
				t.Errorf("output changed:\n got %q\nwant %q", got[c.name], c.want)
			}
		})
	}
	// The invariant the cases instantiate: --cascade adds exactly the line
	// naming the deleted rows to the text summary and nothing to the answer.
	if strings.Replace(got["text cascade"], cascadeLine, "", 1) != got["text"] {
		t.Errorf("--cascade changed more than its Deleted: line:\n%q\n%q", got["text cascade"], got["text"])
	}
	if got["json cascade"] != got["json"] {
		t.Errorf("--cascade changed the --json answer:\n%q\n%q", got["json cascade"], got["json"])
	}
}
