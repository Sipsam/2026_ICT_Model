import assert from "node:assert/strict";
import test from "node:test";
import {
  boxPercent, buildFilters, dailySeries, defaultDateRange, detectionRows, inspectionRecord, pageParams,
} from "../src/ict_pipeline/api/dashboard/client.mjs";

const dates = {from: "2026-10-30", through: "2026-11-02"};
const id = "00000000-0000-4000-8000-000000000001";

test("dashboard defaults to production data and the last 30 UTC dates", () => {
  assert.deepEqual(defaultDateRange(new Date("2026-01-01T00:12:00Z")), ["2025-12-03", "2026-01-01"]);
  assert.deepEqual(Object.fromEntries(buildFilters(dates)), {
    mode: "real", created_from: "2026-10-30T00:00:00.000Z", created_before: "2026-11-03T00:00:00.000Z",
  });
});

test("history, overview and exports can share filters without sharing pagination", () => {
  const filters = buildFilters({
    ...dates, mode: "mock", inspection_id: id, model_id: "mock-defect",
    defect_class: "cold_solder", status: "review_required",
  });
  const page = pageParams(filters, {limit: 25, offset: 50, order: "oldest"});
  const exported = new URLSearchParams(page);
  exported.set("format", "csv");
  assert.equal(page.get("mode"), "mock");
  assert.equal(page.get("inspection_id"), id);
  assert.equal(page.get("defect_class"), "cold_solder");
  assert.equal(page.get("offset"), "50");
  assert.equal(page.get("order"), "oldest");
  assert.equal(page.get("format"), null);
  assert.equal(filters.get("offset"), null);
  assert.throws(() => buildFilters({...dates, inspection_id: "../../image"}), /full inspection UUID/);
  assert.throws(() => buildFilters({...dates, from: "2026-02-30"}), /valid UTC date range/);
  assert.throws(() => buildFilters({...dates, through: "2026-01-01"}), /valid UTC date range/);
  assert.throws(() => buildFilters({...dates, from: "2025-01-01"}), /366 days/);
});

test("daily trend keeps UTC day spacing, and empty data yields no chart points", () => {
  const filters = buildFilters(dates);
  assert.deepEqual(dailySeries([], filters), []);
  assert.deepEqual(dailySeries([
    {date: "2026-10-30", inspection_count: 4}, {date: "2026-11-02", inspection_count: 2},
  ], filters), [
    {date: "2026-10-30", count: 4}, {date: "2026-10-31", count: 0},
    {date: "2026-11-01", count: 0}, {date: "2026-11-02", count: 2},
  ]);
  assert.throws(() => dailySeries([{date: "2026-11-03", inspection_count: 1}], filters), /invalid or unsupported/);
});

test("box percentages use original dimensions and align at different display sizes", () => {
  const examples = [
    {width: 1280, height: 720},
    {width: 480, height: 960},
  ];
  for (const image of examples) {
    const {width, height} = image;
    const rect = boxPercent([width / 8, height / 4, width * .75, height * .75], image);
    assert.deepEqual(rect, {left: 12.5, top: 25, width: 62.5, height: 50});
    for (const scale of [1, 0.5, 0.125]) {
      assert.equal(rect.left * width * scale / 100, width * scale / 8);
      assert.equal(rect.height * height * scale / 100, height * scale / 2);
    }
  }
});

test("detection rendering keeps overlapping valid boxes and rejects malformed geometry", () => {
  const image = {width: 40, height: 80};
  const detection = {class_id: 1, class_name: "cold_solder", bbox_xyxy: [10, 20, 30, 60], confidence: 0.95};
  const rows = detectionRows([
    detection,
    {...detection, class_id: 3, class_name: "short"},
    {...detection, class_name: "<script>"},
    {...detection, bbox_xyxy: [0, 0, 41, 20]},
    {...detection, bbox_xyxy: [0, 0, Infinity, 20]},
    {...detection, bbox_xyxy: [0, 0, 0, 20]},
    {...detection, confidence: "0.95"},
  ], image);
  assert.deepEqual(rows.map(row => [row.name, row.confidence, row.rect]), [
    ["cold_solder", .95, {left: 25, top: 25, width: 50, height: 50}],
    ["short", .95, {left: 25, top: 25, width: 50, height: 50}],
  ]);
  assert.deepEqual(detectionRows([detection], null), []);
});

test("record validation preserves versions and treats identifiers as opaque text", () => {
  const record = {
    schema_version: 1, inspection_id: id, model_id: "mock-defect", created_at: "2026-10-30T17:54:00Z",
    result: {
      schema_version: 1, outcome: "review_required", reason: "simulated_prediction",
      request_id: "<img src=x>", source_id: null, provenance: {mode: "mock"}, detections: [],
    },
  };
  assert.equal(inspectionRecord(record).result.request_id, "<img src=x>");
  assert.throws(() => inspectionRecord({...record, inspection_id: "../private"}), /invalid or unsupported/);
  assert.throws(() => inspectionRecord({...record, result: {...record.result, schema_version: 2}}), /invalid or unsupported/);
  assert.throws(() => inspectionRecord({...record, result: {...record.result, detections: {}}}), /invalid or unsupported/);
});
