import assert from "node:assert/strict";
import test from "node:test";
import {
  APIError, api,
  boxPercent, buildFilters, buildModelRequest, dailySeries, datetimeLocalToISO, defaultDateRange,
  detectionRows, inspectionEndpoint, inspectionRecord, modelCatalog, pageParams, productEvidence, selectDetectionRecord,
} from "../src/ict_pipeline/api/dashboard/client.mjs";

const dates = {from: "2026-10-30", through: "2026-11-02"};
const id = "00000000-0000-4000-8000-000000000001";

test("model catalog accepts schema-less responses and validates mixed availability", () => {
  const models = modelCatalog({models: [
    {model_id: "faster-rcnn-final", mode: "real", model: "faster_rcnn", available: true},
    {model_id: "broken-model", mode: "real", available: false},
  ]});
  assert.deepEqual(models.map(model => [model.model_id, model.available]), [
    ["faster-rcnn-final", true], ["broken-model", false],
  ]);
  assert.deepEqual(modelCatalog({models: []}), []);
  assert.throws(() => modelCatalog({}), /invalid or unsupported/);
  assert.throws(() => modelCatalog({models: {}}), /invalid or unsupported/);
  assert.throws(() => modelCatalog({models: [{model_id: "missing-fields"}]}), /invalid or unsupported/);
  assert.throws(() => modelCatalog({models: [{model_id: "", mode: "real", available: true}]}), /invalid or unsupported/);
});

test("model catalog API preserves bearer tokens and HTTP error handling", async () => {
  globalThis.window = {location: {origin: "https://dashboard.test"}};
  const calls = [];
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async (url, options) => {
      calls.push({url: url.toString(), options});
      return {ok: true, status: 200, json: async () => ({models: []})};
    };
    const response = await api("/v1/models", {token: "catalog-token"});
    assert.deepEqual(modelCatalog(await response.json()), []);
    assert.equal(calls[0].url, "https://dashboard.test/v1/models");
    assert.equal(calls[0].options.headers.Authorization, "Bearer catalog-token");

    globalThis.fetch = async () => ({
      ok: false, status: 503,
      json: async () => ({error: {code: "model_unavailable"}}),
    });
    await assert.rejects(
      api("/v1/models", {token: "catalog-token"}),
      error => error instanceof APIError && error.status === 503
        && error.code === "model_unavailable",
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("product evidence distinguishes pairing, fallback, untrained scores, and missing inputs", () => {
  const product = {
    schema_version: 1, input_mode: "paired", branch: "image", sensor_status: "insufficient_steps",
    weight_status: "untrained", alignment: "verified_receipt", threshold: 0.5, score_kind: "uncalibrated_sigmoid",
    scores: [{name: "any_target_defect", score: 0.79, threshold_crossed: true}],
  };
  const view = productEvidence(product);
  assert.equal(view.input, "Paired image and sensor");
  assert.equal(view.branch, "Image");
  assert.equal(view.sensor, "Insufficient observed steps");
  assert.equal(view.alignment, "Verified pairing receipt");
  assert.equal(view.status, "Untrained weights");
  assert.equal(view.scoreLabel, "Uncalibrated scores");
  assert.equal(buildFilters({...dates, modality: "sensor"}).get("modality"), "sensor");
  const missing = productEvidence({...product, input_mode: "sensor", branch: null,
    weight_status: "unavailable", alignment: "not_applicable", sensor_status: "no_observations", scores: []});
  assert.equal(missing.branch, "Unavailable");
  assert.deepEqual(missing.scores, []);
  assert.throws(() => productEvidence({...product, scores: [{name: "any_target_defect", score: NaN}]}), /unsupported/);
});

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

test("inspection request payloads vary by model input kind", () => {
  const id = "00000000-0000-4000-8000-000000000002";
  const detector = buildModelRequest("faster-rcnn-final", {input_kind: null}, {id});
  assert.deepEqual(detector, {
    schema_version: 1,
    model_id: "faster-rcnn-final",
    request_id: "dashboard-" + id,
    source_id: "dashboard-camera",
  });
  const image = buildModelRequest("fpn-image-only", {input_kind: "multimodal"}, {id});
  assert.equal(image.schema_version, 1);
  assert.equal(image.sample_id, "dashboard-" + id);
  const sensorEvidence = {
    capture_timestamp: "2026-11-23T12:26:00.000Z",
    sensor_evidence: {source_id: "dashboard-sensor", sequence: {channels: ["temperature"], units: ["°C"], steps: []}},
  };
  const sensor = buildModelRequest("fpn-sensor-concat", {input_kind: "multimodal"}, {id, sensor: sensorEvidence});
  assert.equal(sensor.sample_id, "dashboard-" + id);
  assert.equal(sensor.capture_timestamp, sensorEvidence.capture_timestamp);
  assert.deepEqual(sensor.sensor_evidence, sensorEvidence.sensor_evidence);
  assert.equal("sensor_evidence" in detector, false);
  assert.equal(inspectionEndpoint({input_kind: null}), "/v1/inspections");
  assert.equal(inspectionEndpoint({input_kind: "multimodal"}), "/v1/inspections/multimodal");
});

test("comparison selects the actual detector record and preserves zero detections", () => {
  const imageResult = {image: {width: 40, height: 80}, detections: [], multimodal: {schema_version: 1}};
  const detectorResult = {image: {width: 40, height: 80}, detections: []};
  const records = [{model_id: "fpn-image-only", result: imageResult},
                   {model_id: "faster-rcnn-final", result: detectorResult}];
  assert.equal(selectDetectionRecord(records)?.model_id, "faster-rcnn-final");
  assert.equal(selectDetectionRecord(records).result.detections.length, 0);
  assert.equal(selectDetectionRecord([{model_id: "fpn-image-only", result: imageResult}]), null);
  assert.equal(selectDetectionRecord([]), null);
});

test("datetime-local values convert to valid UTC ISO timestamps and reject invalid values", () => {
  const iso = datetimeLocalToISO("2026-11-23T21:26");
  assert.match(iso, /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00\.000Z$/);
  const expected = new Date(2026, 10, 23, 21, 26).toISOString();
  assert.equal(iso, expected);
  assert.equal(datetimeLocalToISO("2026-11-23T21:26:30.123"), new Date(2026, 10, 23, 21, 26, 30, 123).toISOString());
  assert.throws(() => datetimeLocalToISO("2026-02-30T12:00"), /valid local/);
  assert.throws(() => datetimeLocalToISO("not-a-date"), /valid local/);
  assert.throws(() => datetimeLocalToISO("2026-11-23T24:00"), /valid local/);
});
