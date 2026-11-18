// Same-origin API client and view data transformations.
export const DAY = 86_400_000;
export const MAX_DAYS = 366;
export const UUID4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export const STATUSES = new Map([
  ["defect_detected", "Defect detected"], ["review_required", "Review required"],
  ["inference_failed", "Inference failed"], ["pass", "Pass"],
]);
export const CLASSES = ["cold_solder", "misalignment", "short", "missing_component", "tombstone"];
export const isCount = value => Number.isSafeInteger(value) && value >= 0;
const textOrNull = value => value === null || typeof value === "string";
const dateText = day => new Date(day).toISOString().slice(0, 10);

export function utcDay(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value) || value.startsWith("0000")) return NaN;
  const day = Date.parse(value + "T00:00:00Z");
  return Number.isFinite(day) && dateText(day) === value ? day : NaN;
}

export function defaultDateRange(now = new Date()) {
  const today = utcDay(dateText(now));
  return [dateText(today - 29 * DAY), dateText(today)];
}

export function buildFilters(values) {
  const from = utcDay(values.from);
  const through = utcDay(values.through);
  if (!Number.isFinite(from) || !Number.isFinite(through) || through < from) {
    throw new Error("Choose a valid UTC date range.");
  }
  const before = through + DAY;
  if (before - from > MAX_DAYS * DAY || dateText(before).startsWith("+")) {
    throw new Error("Choose a UTC date range of at most 366 days ending before year 10000.");
  }
  const id = values.inspection_id?.trim();
  if (id && !UUID4.test(id)) throw new Error("Enter the full inspection UUID to search by ID.");
  const params = new URLSearchParams({
    mode: values.mode === "mock" ? "mock" : "real",
    created_from: new Date(from).toISOString(),
    created_before: new Date(before).toISOString(),
  });
  for (const field of ["status", "defect_class", "model", "model_id"]) {
    if (values[field]) params.set(field, values[field]);
  }
  if (id) params.set("inspection_id", id);
  return params;
}

export function pageParams(filters, {limit, offset, order}) {
  const params = new URLSearchParams(filters);
  params.set("limit", limit);
  params.set("offset", offset);
  params.set("order", order);
  return params;
}

export class APIError extends Error {
  constructor(status, code) {
    super({
      401: "API access requires a valid bearer token. Enter it under API access and apply filters.",
      403: "The API denied access.",
      404: "The requested record or image is unavailable.",
      422: "The API rejected the request. Check the active filters.",
      429: "The API is busy. Try again shortly.",
      503: "The inspection store or API feature is unavailable.",
    }[status] ?? "The API request failed.");
    this.status = status;
    this.code = code;
  }
}

export async function api(path, {params, token, signal} = {}) {
  const url = new URL(path, window.location.origin);
  if (url.origin !== window.location.origin || !url.pathname.startsWith("/v1/")) {
    throw new Error("Invalid inspection API endpoint.");
  }
  if (params) url.search = params.toString();
  const headers = token ? {Authorization: "Bearer " + token} : {};
  const response = await fetch(url, {headers, signal, redirect: "error", cache: "no-store"});
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new APIError(response.status, body?.error?.code);
  }
  return response;
}

export async function json(path, options) {
  const response = await api(path, options);
  const data = await response.json().catch(() => null);
  requirePayload(data?.schema_version === 1);
  return data;
}

export function requirePayload(condition) {
  if (!condition) throw new Error("The API returned an invalid or unsupported response.");
}

export function inspectionRecord(record) {
  const result = record?.result;
  requirePayload(record?.schema_version === 1 && result?.schema_version === 1
    && typeof record.inspection_id === "string" && UUID4.test(record.inspection_id)
    && typeof record.model_id === "string" && typeof record.created_at === "string"
    && Number.isFinite(Date.parse(record.created_at)) && Array.isArray(result.detections)
    && STATUSES.has(result.outcome) && typeof result.reason === "string"
    && textOrNull(result.request_id) && textOrNull(result.source_id)
    && ["real", "mock"].includes(result.provenance?.mode));
  return record;
}

export function imageSize(image) {
  return isCount(image?.width) && isCount(image?.height) && image.width > 0 && image.height > 0;
}

export function boxPercent(box, image) {
  if (!imageSize(image) || !Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) return null;
  const [x1, y1, x2, y2] = box;
  if (x1 < 0 || y1 < 0 || x1 >= x2 || y1 >= y2 || x2 > image.width || y2 > image.height) return null;
  return {left: x1 / image.width * 100, top: y1 / image.height * 100,
          width: (x2 - x1) / image.width * 100, height: (y2 - y1) / image.height * 100};
}

export function detectionRows(detections, image) {
  return detections.flatMap((detection, index) => {
    const name = detection?.class_name;
    const confidence = detection?.confidence;
    const rect = boxPercent(detection?.bbox_xyxy, image);
    if (!Number.isInteger(detection?.class_id) || CLASSES[detection.class_id - 1] !== name || !CLASSES.includes(name)
        || !Number.isFinite(confidence)
        || confidence < 0 || confidence > 1 || !rect) return [];
    return [{index, name, confidence, rect, coordinates: detection.bbox_xyxy}];
  });
}

export function dailySeries(items, params) {
  const start = Date.parse(params.get("created_from"));
  const end = Date.parse(params.get("created_before"));
  requirePayload(Array.isArray(items) && end > start && end - start <= MAX_DAYS * DAY);
  const rows = new Map();
  for (const row of items) {
    const day = utcDay(row?.date);
    requirePayload(day >= start && day < end && !rows.has(day) && isCount(row.inspection_count));
    rows.set(day, row.inspection_count);
  }
  if (!rows.size) return [];
  const series = [];
  for (let day = start; day < end; day += DAY) {
    series.push({date: dateText(day), count: rows.get(day) ?? 0});
  }
  return series;
}
