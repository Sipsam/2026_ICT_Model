import {
  APIError, CLASSES, STATUSES, api, buildFilters, dailySeries, defaultDateRange,
  buildModelRequest, datetimeLocalToISO, detectionRows, imageSize, inspectionRecord, isCount, json,
  inspectionEndpoint, pageParams, productEvidence, requirePayload, inspectMultipart, selectDetectionRecord,
} from "./client.mjs";

const byId = id => document.getElementById(id);
const form = byId("filters");
const dialog = byId("detail");
const detailContent = byId("detail-content");
const number = new Intl.NumberFormat();
const integer = value => number.format(value);
const utcTime = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium", timeStyle: "medium", timeZone: "UTC",
});
const statusButtons = ["apply", "refresh", "order", "limit"].map(byId);
const colors = ["#4dfada", "#ffd06c", "#ff9f94", "#9bd7ff", "#e2b7ff"];
const state = {
  filters: null, token: "", order: "newest", limit: 25, offset: 0, page: null,
  loading: false, exporting: false, detailRequest: null, detailTrigger: null, imageURL: null, timer: null,
  selectedFile: null, previewURL: null, models: [], inspectionBusy: false, comparisonRecords: [],
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showNotice(id, message = "", error = false) {
  const node = byId(id);
  node.textContent = message;
  node.classList.toggle("error", error);
  node.hidden = !message;
}

function errorMessage(error) {
  return error instanceof TypeError || error instanceof DOMException
    ? "The dashboard could not reach the API. Check the server and try again."
    : error.message;
}

function options(params, signal) {
  const timeout = AbortSignal.timeout(20_000);
  return {params, token: state.token, signal: signal ? AbortSignal.any([signal, timeout]) : timeout};
}

function barList(target, rows, empty, chunkSize = 12) {
  target.replaceChildren();
  if (!rows.length) {
    target.append(element("p", empty, "empty"));
    return;
  }
  const maximum = rows.reduce((max, row) => Math.max(max, row.count), 1);
  const list = element("ul", undefined, "bars");
  const more = element("button", "Show more", "more");
  more.type = "button";
  let visible = 0;
  function appendRows(focus = false) {
    let first;
    for (const row of rows.slice(visible, visible + chunkSize)) {
      const item = element("li");
      const label = element("div", undefined, "bar-text");
      label.append(element("span", row.label), element("strong", row.note ?? integer(row.count)));
      const track = element("span", undefined, "track");
      track.setAttribute("aria-hidden", "true");
      const fill = element("span");
      fill.style.width = 100 * row.count / maximum + "%";
      track.append(fill);
      item.append(label, track);
      list.append(item);
      first ??= item;
    }
    visible += chunkSize;
    if (focus && first) {
      first.tabIndex = -1;
      first.focus();
    }
    more.hidden = visible >= rows.length;
  }
  more.addEventListener("click", () => appendRows(true));
  target.append(list, more);
  appendRows();
}

function renderTrend(rows) {
  const node = byId("trend");
  node.replaceChildren();
  if (!rows.length) {
    node.append(element("p", "No inspections in this date range.", "empty"));
    return;
  }
  const max = Math.max(1, ...rows.map(row => row.count));
  const scroll = element("div", undefined, "trend-scroll");
  const plot = element("ol", undefined, "trend");
  plot.setAttribute("aria-hidden", "true");
  const tickStep = Math.max(1, Math.ceil(rows.length / 5));
  rows.forEach(({date, count}, index) => {
    const day = element("li", undefined, "day");
    day.title = date + " · " + integer(count) + " inspections";
    const track = element("span", undefined, "plot");
    const bar = element("span", undefined, "column");
    bar.style.height = 100 * count / max + "%";
    track.append(bar);
    day.append(track);
    if (index % tickStep === 0) day.append(element("span", date.slice(5), "tick"));
    plot.append(day);
  });
  scroll.append(plot);
  if (rows.length > 20) {
    scroll.tabIndex = 0;
    scroll.setAttribute("aria-label", "Scrollable daily trend");
  }
  const values = element("details", undefined, "trend-values");
  values.append(element("summary", "View daily counts"));
  values.addEventListener("toggle", () => {
    if (!values.open || values.children.length > 1) return;
    const list = element("ul");
    for (const {date, count} of rows) list.append(element("li", date + " · " + integer(count) + " inspections"));
    values.append(list);
  });
  node.append(scroll, values);
}

function countEntries(values) {
  requirePayload(values && typeof values === "object" && !Array.isArray(values));
  const entries = Object.entries(values);
  requirePayload(entries.every(([, count]) => isCount(count)));
  return entries;
}

async function loadOverview() {
  const params = state.filters;
  const mode = params.get("mode");
  try {
    // Keep one API request in flight even when the server is configured with capacity one.
    const summary = await json("/v1/statistics/summary", options(params));
    requirePayload(summary.mode === mode && isCount(summary.total_inspections) && isCount(summary.total_detections));
    const statuses = countEntries(summary.by_status);
    requirePayload(statuses.every(([key]) => STATUSES.has(key)));
    const models = countEntries(summary.by_model).sort(([a, ac], [b, bc]) => bc - ac || a.localeCompare(b));
    const defects = await json("/v1/statistics/defects", options(params));
    requirePayload(defects.mode === mode && Array.isArray(defects.items) && defects.items.every(row =>
      CLASSES.includes(row?.class_name) && isCount(row.inspection_count) && isCount(row.detection_count)));
    const trends = await json("/v1/statistics/trends", options(params));
    requirePayload(trends.mode === mode);
    const days = dailySeries(trends.items, params);

    byId("inspection-total").textContent = integer(summary.total_inspections);
    byId("detection-total").textContent = integer(summary.total_detections);
    barList(byId("statuses"), statuses.map(([key, count]) => ({label: STATUSES.get(key), count})), "No inspection outcomes.");
    barList(byId("defects"), defects.items.map(row => ({
      label: row.class_name, count: row.detection_count,
      note: integer(row.detection_count) + " detections · " + integer(row.inspection_count) + " inspections",
    })), "No detections.");
    barList(byId("models"), models.map(([label, count]) => ({label, count})), "No detector model counts.");
    renderTrend(days);
    byId("charts").hidden = false;
    byId("updated").textContent = "Updated " + utcTime.format(new Date()) + " UTC";
    showNotice("overview-feedback", summary.total_inspections === 0 ? "No inspections match these filters." : "");
  } catch (error) {
    showNotice("overview-feedback", errorMessage(error), true);
  }
}

function statusBadge(status) {
  return element("span", STATUSES.get(status), "badge " + status);
}

function historyRow(record) {
  const row = element("tr");
  const result = record.result;
  const timestamp = element("time", utcTime.format(new Date(record.created_at)));
  timestamp.dateTime = record.created_at;
  const created = element("td");
  created.append(timestamp);
  const source = element("td", result.request_id ?? "—");
  source.append(element("span", result.source_id ?? "—", "muted"));
  const status = element("td");
  status.append(statusBadge(result.outcome), element("span", result.reason, "muted"));
  const model = element("td", record.model_id);
  if (result.provenance.mode === "mock") model.append(element("span", "Simulated", "muted"));
  else model.append(element("span", result.multimodal ? productEvidence(result.multimodal).status :
    result.provenance.model ?? "—", "muted"));
  const details = element("td");
  const open = element("button", "View", "record-button");
  open.type = "button";
  open.setAttribute("aria-label", "View inspection " + (result.request_id ?? record.inspection_id));
  open.addEventListener("click", () => openDetail(record.inspection_id, open));
  details.append(open);
  row.append(created, source, status, model, element("td", integer(result.detections.length), "numeric"), details);
  return row;
}

function queryPage(offset) {
  return pageParams(state.filters, {order: state.order, limit: state.limit, offset});
}

async function fetchPage(offset) {
  const page = await json("/v1/inspections", options(queryPage(offset)));
  requirePayload(isCount(page.total) && page.limit === state.limit && page.offset === offset
    && Array.isArray(page.items) && page.items.length === Math.max(0, Math.min(page.limit, page.total - page.offset)));
  return page;
}

async function loadHistory() {
  try {
    let page = await fetchPage(state.offset);
    if (page.offset > 0 && !page.items.length) {
      state.offset = Math.max(0, Math.floor((page.total - 1) / state.limit) * state.limit);
      page = await fetchPage(state.offset);
    }
    const rows = page.items.map(inspectionRecord);
    requirePayload(rows.every(record => record.result.provenance.mode === state.filters.get("mode")));
    byId("rows").replaceChildren(...rows.map(historyRow));
    byId("table-wrap").hidden = !rows.length;
    byId("page-version").textContent = "API schema v" + page.schema_version;
    byId("page-label").textContent = rows.length
      ? `${integer(page.offset + 1)}–${integer(page.offset + rows.length)} of ${integer(page.total)} inspections`
      : `Showing 0 of ${integer(page.total)} inspections`;
    showNotice("history-feedback", rows.length ? "" : page.total
      ? "This page is now empty. Go to the previous page or refresh."
      : "No inspections match these filters.");
    state.page = page;
  } catch (error) {
    showNotice("history-feedback", errorMessage(error), true);
  }
}

function syncControls() {
  const busy = state.loading || state.exporting;
  const page = state.page;
  for (const button of statusButtons) button.disabled = busy;
  byId("previous").disabled = busy || !page || page.offset === 0;
  byId("next").disabled = busy || !page || page.offset + page.limit >= page.total;
  byId("csv").disabled = byId("json").disabled = busy || !page?.items.length;
}

async function load({overview = false, focus = false} = {}) {
  if (state.loading || state.exporting || dialog.open) return;
  state.loading = true;
  state.page = null;
  syncControls();
  byId("table-wrap").hidden = true;
  byId("page-label").textContent = "";
  byId("page-version").textContent = "";
  showNotice("export-feedback");
  showNotice("history-feedback", "Loading history…");
  if (overview) {
    byId("charts").hidden = true;
    byId("updated").textContent = "";
    showNotice("overview-feedback", "Loading statistics…");
  }
  try {
    if (overview) await loadOverview();
    await loadHistory();
    if (focus) {
      const result = byId("page-label").textContent ? byId("page-label") : byId("history-feedback");
      result.tabIndex = -1;
      result.focus();
    }
  } finally {
    state.loading = false;
    syncControls();
    scheduleRefresh();
  }
}

function applyFilters() {
  let filters;
  try {
    filters = buildFilters(Object.fromEntries(new FormData(form)));
  } catch (error) {
    showNotice("filter-feedback", errorMessage(error), true);
    return;
  }
  showNotice("filter-feedback");
  state.filters = filters;
  state.token = byId("token").value.trim();
  state.offset = 0;
  const modeLabel = byId("mode-label");
  const mock = filters.get("mode") === "mock";
  modeLabel.textContent = mock ? "Simulated inspections" : "Real inspections";
  modeLabel.classList.toggle("mock", mock);
  void loadModels();
  void load({overview: true});
}

function scheduleRefresh() {
  clearTimeout(state.timer);
  const seconds = Number(byId("interval").value);
  if (![60, 120, 300].includes(seconds) || document.hidden || dialog.open || state.loading || state.exporting) return;
  state.timer = setTimeout(() => {
    if (!document.hidden) void load({overview: true});
  }, seconds * 1000);
}

async function exportPage(format) {
  if (!state.page || state.loading || state.exporting) return;
  state.exporting = true;
  syncControls();
  scheduleRefresh();
  showNotice("export-feedback", "Preparing " + format.toUpperCase() + " export…");
  try {
    const params = queryPage(state.page.offset);
    params.set("format", format);
    const response = await api("/v1/inspections/export", options(params));
    const blob = await response.blob();
    requirePayload(blob.type.startsWith(format === "csv" ? "text/csv" : "application/json"));
    const url = URL.createObjectURL(blob);
    const link = element("a");
    link.href = url;
    link.download = "inspection-history." + format;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showNotice("export-feedback", "Export ready.");
  } catch (error) {
    showNotice("export-feedback", errorMessage(error), true);
  } finally {
    state.exporting = false;
    syncControls();
    scheduleRefresh();
  }
}

function facts(entries) {
  const list = element("dl", undefined, "facts");
  for (const [label, value] of entries) {
    if (value === null || value === undefined) continue;
    list.append(element("dt", label), element("dd", value));
  }
  return list;
}

function recordFacts(record) {
  const {result} = record;
  const {provenance, image, thresholds} = result;
  return facts([
    ["Inspection ID", record.inspection_id], ["Created (UTC)", utcTime.format(new Date(record.created_at))],
    ["Schema versions", `Record ${record.schema_version} · Result ${result.schema_version}`],
    ["Request ID", result.request_id], ["Source ID", result.source_id], ["Sensor source ID", result.sensor_source_id],
    ["Model ID", record.model_id], ["Detector family", provenance.model], ["Mock scenario", provenance.mock_scenario],
    ["Label policy", provenance.label_policy], ["Checkpoint", provenance.checkpoint_kind],
    ["Checkpoint fingerprint", provenance.checkpoint_fingerprint], ["Run fingerprint", provenance.run_metadata_fingerprint],
    ["Device", provenance.device], ["Inference image size", provenance.image_size],
    ["Confidence threshold", thresholds?.confidence], ["NMS IoU threshold", thresholds?.nms_iou],
    ["Original image", imageSize(image) ? `${image.width} × ${image.height} px · ${image.format}` : null],
    ["Image SHA-256", image?.sha256], ["Verification reference", result.verification_reference],
  ]);
}

function renderProduct(product) {
  const panel = element("section");
  const view = productEvidence(product);
  panel.append(element("h3", "Product prediction"), facts([
    ["Sample ID", product.sample_id], ["Input", view.input], ["Scored branch", view.branch],
    ["Image", product.input_mode === "sensor" ? "Absent" : "Available"],
    ["Sensor", view.sensor], ["Prediction status", view.status], ["Alignment", view.alignment],
    ["Capture time", product.capture_timestamp], ["Pairing fingerprint", product.pairing_fingerprint],
    ["Sensor fingerprint", product.sensor_fingerprint], ["Threshold", product.threshold],
  ]));
  if (view.scores.length) {
    const list = element("ul");
    for (const row of view.scores) {
      list.append(element("li", `${row.name} ${row.score.toFixed(4)}${row.threshold_crossed ? " (threshold crossed)" : ""}`));
    }
    panel.append(element("h4", view.scoreLabel), list);
  } else {
    panel.append(element("p", "No usable product scores.", "muted"));
  }
  return panel;
}

const detectionLabel = row => row.name + " · " + (row.confidence * 100).toFixed(1) + "%";

function renderImage(container, url, image, rows, boxes) {
  const figure = element("figure");
  const stage = element("div", undefined, "image-stage");
  stage.style.aspectRatio = image.width + " / " + image.height;
  stage.style.maxWidth = Math.min(840, 520 * image.width / image.height) + "px";
  const original = element("img");
  original.alt = "Original inspection image";
  original.src = url;
  original.addEventListener("error", () => container.replaceChildren(
    element("p", "The browser could not display the original image.", "notice"),
  ));
  stage.append(original);
  const overlay = element("div");
  overlay.setAttribute("aria-hidden", "true");
  for (const row of rows) {
    const box = element("div", undefined, "box");
    box.style.setProperty("--box", colors[CLASSES.indexOf(row.name)]);
    for (const [key, value] of Object.entries(row.rect)) box.style[key] = value + "%";
    box.append(element("span", detectionLabel(row)));
    overlay.append(box);
    boxes.set(row.index, box);
  }
  stage.append(overlay);
  figure.append(stage, element("figcaption", "Boxes use original-image pixels. Focus a detection below to highlight its box."));
  container.replaceChildren(figure);
}

function renderDetections(container, rows, total, boxes) {
  container.append(element("h3", "Detections · " + integer(total)));
  if (rows.length < total) {
    container.append(element("p", "Some detections have invalid coordinates or labels and could not be displayed.", "notice error"));
  }
  if (total === 0) container.append(element("p", "No retained detections.", "muted"));
  if (rows.length === 0) return;
  const list = element("ol", undefined, "detection-list");
  const more = element("button", "Show more detections", "more");
  more.type = "button";
  let shown = 0;
  function appendRows(focus = false) {
    let first;
    for (const row of rows.slice(shown, shown + 50)) {
      const item = element("li");
      const description = detectionLabel(row) + " · [" + row.coordinates.join(", ") + "] px";
      const highlight = element("button", description);
      highlight.type = "button";
      const toggle = enabled => boxes.get(row.index)?.classList.toggle("active", enabled);
      highlight.addEventListener("focus", () => toggle(true));
      highlight.addEventListener("blur", () => toggle(false));
      highlight.addEventListener("mouseenter", () => toggle(true));
      highlight.addEventListener("mouseleave", () => toggle(document.activeElement === highlight));
      highlight.addEventListener("click", () => {
        const box = boxes.get(row.index);
        if (box) {
          box.classList.add("active");
          box.scrollIntoView({block: "center", inline: "nearest"});
        }
      });
      item.append(highlight);
      list.append(item);
      first ??= highlight;
    }
    shown += 50;
    if (focus) first?.focus();
    more.hidden = shown >= rows.length;
  }
  more.addEventListener("click", () => appendRows(true));
  container.append(list, more);
  appendRows();
}

async function openDetail(id, trigger) {
  if (state.loading || state.exporting || dialog.open) return;
  const controller = new AbortController();
  state.detailRequest = controller;
  state.detailTrigger = trigger;
  detailContent.replaceChildren(element("p", "Loading inspection…", "notice"));
  dialog.showModal();
  scheduleRefresh();
  const path = "/v1/inspections/" + encodeURIComponent(id);
  try {
    const record = inspectionRecord(await json(path, options(undefined, controller.signal)));
    requirePayload(record.inspection_id === id);
    if (controller.signal.aborted) return;
    const result = record.result;
    const heading = element("div", undefined, "detail-heading");
    heading.append(statusBadge(result.outcome), element("span", result.reason, "muted"));
    if (result.provenance.mode === "mock") heading.append(element("span", "Simulated prediction", "badge mock"));
    const media = element("div");
    const detectionPanel = element("div");
    const boxes = new Map();
    const rows = detectionRows(result.detections, result.image);
    renderDetections(detectionPanel, rows, result.detections.length, boxes);
    const raw = element("details");
    raw.append(element("summary", "View versioned JSON"));
    raw.addEventListener("toggle", () => {
      if (raw.open && raw.children.length === 1) raw.append(element("pre", JSON.stringify(record, null, 2)));
    });
    const productPanels = result.multimodal ? [renderProduct(result.multimodal)] : [];
    detailContent.replaceChildren(heading, ...productPanels, media, detectionPanel, recordFacts(record), raw);
    if (!imageSize(result.image)) {
      media.append(element("p", "No original image is recorded for this inspection.", "notice"));
      return;
    }
    media.append(element("p", "Loading original image…", "notice"));
    try {
      const response = await api(path + "/image", options(undefined, controller.signal));
      const blob = await response.blob();
      requirePayload(blob.type === "image/jpeg" || blob.type === "image/png");
      if (controller.signal.aborted) return;
      state.imageURL = URL.createObjectURL(blob);
      renderImage(media, state.imageURL, result.image, rows, boxes);
    } catch (error) {
      if (controller.signal.aborted) return;
      const missing = error instanceof APIError && error.status === 404;
      media.replaceChildren(element("p", missing ? "Original image unavailable. Detection coordinates are listed below."
        : errorMessage(error), missing ? "notice" : "notice error"));
    }
  } catch (error) {
    if (!controller.signal.aborted) detailContent.replaceChildren(element("p", errorMessage(error), "notice error"));
  }
}

async function loadModels() {
  try {
    const response = await json("/v1/models", options());
    requirePayload(Array.isArray(response.models));
    state.models = response.models;
    const select = byId("model-select");
    select.replaceChildren();
    for (const model of response.models) {
      const option = element("option", model.model_id + (model.available ? "" : " — unavailable"), " ");
      option.value = model.model_id;
      option.disabled = !model.available;
      select.append(option);
    }
    if (!response.models.some(model => model.available)) {
      select.innerHTML = '<option value="">No models available</option>';
    }
    byId("workspace-status").textContent = `${response.models.filter(model => model.available).length} models available`;
    updateModelSelection();
  } catch (error) {
    byId("workspace-status").textContent = "Model catalog unavailable";
    showNotice("inspection-feedback", errorMessage(error), true);
  }
}

function updateModelSelection() {
  const model = state.models.find(row => row.model_id === byId("model-select").value);
  const sensor = byId("sensor-panel");
  sensor.hidden = !(model?.input_kind === "multimodal" && model.model_id === "fpn-sensor-concat");
  if (model) {
    byId("model-description").textContent = !model.available
      ? "This model is unavailable."
      : model.model_id === "fpn-sensor-concat"
        ? "Sensor fusion; paired verification and capture-time evidence are required for fusion."
        : model.model_id === "fpn-image-only"
          ? "Image-only multilabel classification; sigmoid scores are uncalibrated."
          : model.model_id === "faster-rcnn-final"
            ? "Finalized Faster R-CNN object detection."
            : model.input_kind === "multimodal"
              ? "Multimodal product model. Sensor evidence is optional."
              : "Configured inspection model.";
  }
  updateRunButton();
}

function updateRunButton() {
  const selected = byId("model-select").value;
  byId("run-inspection").disabled = state.inspectionBusy || !state.selectedFile || !selected;
}

async function validateImage(file) {
  if (!file) return null;
  if (!["image/jpeg", "image/png"].includes(file.type)) {
    throw new Error("Choose a JPG or PNG image.");
  }
  if (file.size > 8 * 1024 * 1024) {
    throw new Error("The image exceeds the 8 MiB upload limit.");
  }
  const dimensions = await new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      resolve({width: image.naturalWidth, height: image.naturalHeight, url});
      URL.revokeObjectURL(url);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("The selected file is not a readable JPEG or PNG."));
    };
    image.src = url;
  });
  if (dimensions.width * dimensions.height > 16_000_000
      || dimensions.width > 6000 || dimensions.height > 6000) {
    throw new Error("The image exceeds the server pixel or dimension limit.");
  }
  return dimensions;
}

async function selectFile(file) {
  showNotice("upload-feedback");
  try {
    const dimensions = await validateImage(file);
    if (!dimensions) return;
    if (state.previewURL) URL.revokeObjectURL(state.previewURL);
    state.selectedFile = file;
    state.previewURL = URL.createObjectURL(file);
    byId("image-preview").src = state.previewURL;
    byId("image-name").textContent = file.name;
    byId("image-meta").textContent = `${integer(file.size)} bytes · ${dimensions.width} × ${dimensions.height} px · ${file.type}`;
    byId("image-preview-wrap").hidden = false;
    byId("dropzone").hidden = true;
  } catch (error) {
    state.selectedFile = null;
    showNotice("upload-feedback", error.message, true);
  } finally {
    updateRunButton();
  }
}

function clearSelectedImage() {
  state.selectedFile = null;
  byId("image-file").value = "";
  byId("image-preview-wrap").hidden = true;
  byId("dropzone").hidden = false;
  if (state.previewURL) URL.revokeObjectURL(state.previewURL);
  state.previewURL = null;
  showNotice("upload-feedback");
  updateRunButton();
}

function sensorPayload() {
  const channelIds = ["temperature", "humidity", "vibration", "acceleration", "noise"];
  const units = ["°C", "%", "m/s²", "g", "dB"];
  const values = channelIds.map(name => {
    const value = byId("sensor-" + name).value;
    if (value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  });
  const timestamp = byId("capture-time").value;
  const hasValues = values.some(value => value !== null);
  if (!timestamp || !hasValues) return null;
  let captureTimestamp;
  try {
    captureTimestamp = datetimeLocalToISO(timestamp);
  } catch {
    return null;
  }
  return {
    capture_timestamp: captureTimestamp,
    sensor_evidence: {
      source_id: byId("sensor-source").value.trim() || "dashboard-sensor",
      sequence: {
        channels: channelIds,
        units,
        steps: [{timestamp: captureTimestamp, values}],
      },
    },
  };
}

function modelRequestDetails(modelId) {
  const id = crypto.randomUUID();
  const model = state.models.find(row => row.model_id === modelId);
  const sensor = (model?.input_kind === "multimodal" || modelId === "fpn-sensor-concat")
    && modelId === "fpn-sensor-concat"
    ? sensorPayload() : null;
  return buildModelRequest(modelId, model, {id, sensor});
}

function renderProductScores(container, product) {
  const view = productEvidence(product);
  container.append(element("h3", "Classification scores"));
  container.append(element("p", view.scoreLabel + " · fixed threshold " + product.threshold.toFixed(2), "muted"));
  if (!view.scores.length) {
    container.append(element("p", "No usable product scores.", "muted"));
    return;
  }
  const bars = element("div", undefined, "score-bars");
  for (const row of view.scores) {
    const item = element("div", undefined, "score-row");
    const label = element("div");
    label.append(element("span", row.name), element("strong",
      row.score.toFixed(4) + (row.threshold_crossed ? " · above threshold" : " · below threshold")));
    const track = element("span", undefined, "track");
    const fill = element("span", undefined, row.threshold_crossed ? "score-above" : "");
    fill.style.width = (100 * row.score).toFixed(2) + "%";
    track.append(fill);
    item.append(label, track);
    bars.append(item);
  }
  container.append(bars);
  container.append(element("p",
    "These are uncalibrated sigmoid outputs, not calibrated probabilities.", "muted"));
}

let activeWorkspaceImage = null;

function renderWorkspaceResult(records) {
  state.comparisonRecords = records;
  const content = byId("result-content");
  content.replaceChildren();
  byId("inspection-result").hidden = false;
  const grid = element("div", undefined, "result-grid");
  const media = element("div");
  const analysis = element("div");
  grid.append(media, analysis);
  content.append(grid);
  if (state.previewURL) {
    activeWorkspaceImage = new Image();
    activeWorkspaceImage.onload = () => renderWorkspaceBoxes(media);
    activeWorkspaceImage.src = state.previewURL;
  }
  for (const record of records) {
    const card = element("article", undefined, "model-result");
    const result = record.result;
    card.append(element("h3", record.model_id), statusBadge(result.outcome));
    card.append(element("p", result.reason, "muted"));
    if (result.multimodal) renderProductScores(card, result.multimodal);
    if (result.detections.length) {
      card.append(element("p", `${result.detections.length} retained detection(s)`, "muted"));
    } else if (!result.multimodal) {
      card.append(element("p", "No retained detections.", "muted"));
    }
    analysis.append(card);
  }
  byId("save-result").disabled = records.length === 0;
  renderWorkspaceBoxes(media);
}

function renderWorkspaceBoxes(media) {
  if (!activeWorkspaceImage?.complete || !state.previewURL) return;
  const record = selectDetectionRecord(state.comparisonRecords);
  if (!record) {
    media.replaceChildren(element("p", "No original-image result was returned.", "notice"));
    return;
  }
  const result = record.result;
  const image = result.image;
  const rows = detectionRows(result.detections, image).filter(row =>
    !byId("class-filter").value || row.name === byId("class-filter").value);
  const figure = element("figure");
  const stage = element("div", undefined, "image-stage");
  stage.style.aspectRatio = image.width + " / " + image.height;
  stage.style.maxWidth = "840px";
  const original = element("img");
  original.src = state.previewURL;
  original.alt = "Uploaded PCB with inspection overlay";
  stage.append(original);
  if (byId("box-visibility").value === "show") {
    for (const row of rows) {
      const box = element("div", undefined, "box");
      box.style.setProperty("--box", colors[CLASSES.indexOf(row.name)]);
      for (const [key, value] of Object.entries(row.rect)) box.style[key] = value + "%";
      box.append(element("span", detectionLabel(row)));
      stage.append(box);
    }
  }
  figure.append(stage);
  figure.append(element("figcaption",
    `${rows.length} displayed box(es) · coordinates use the ${image.width} × ${image.height} original pixel grid.`));
  media.replaceChildren(figure);
}

function compareTargets() {
  const available = state.models.filter(model => model.available);
  if (!byId("compare-models").checked) {
    const selected = byId("model-select").value;
    return available.filter(model => model.model_id === selected);
  }
  return available.filter(model => model.model_id === "faster-rcnn-final"
    || model.model_id === "fpn-image-only"
    || (model.model_id === "fpn-sensor-concat" && sensorPayload() !== null));
}

async function runInspection() {
  if (state.inspectionBusy || !state.selectedFile) return;
  const targets = compareTargets();
  if (!targets.length) {
    showNotice("inspection-feedback", "Select an available model.", true);
    return;
  }
  const selectedSensor = targets.find(model => model.model_id === "fpn-sensor-concat");
  if (selectedSensor && sensorPayload() === null) {
    showNotice("inspection-feedback",
      "Sensor Concat requires capture timestamp and at least one observed sensor value for paired evidence.",
      true);
    return;
  }
  state.inspectionBusy = true;
  updateRunButton();
  showNotice("inspection-feedback", `Running ${targets.length} inspection(s)…`);
  byId("workspace-status").textContent = "Inference in progress";
  const records = [];
  const failures = [];
  for (const model of targets) {
    const details = modelRequestDetails(model.model_id);
    try {
      const record = await inspectMultipart(inspectionEndpoint(model), {
        file: state.selectedFile, details, token: state.token,
      });
      records.push(inspectionRecord(record));
    } catch (error) {
      failures.push(`${model.model_id}: ${error.code ? error.message + " (" + error.code + ")" : errorMessage(error)}`);
    }
  }
  state.inspectionBusy = false;
  updateRunButton();
  if (records.length) {
    renderWorkspaceResult(records);
    showNotice("inspection-feedback",
      `${records.length} successful · ${failures.length} failed${failures.length ? ". " + failures.join(" | ") : "."}`);
  } else {
    showNotice("inspection-feedback", failures.join(" | ") || "The inspection failed.", true);
  }
  byId("workspace-status").textContent = records.length ? "Inspection complete" : "Inspection failed";
  void load({overview: true});
}

form.addEventListener("submit", event => {
  event.preventDefault();
  if (!state.loading && !state.exporting) applyFilters();
});
byId("choose-image").addEventListener("click", () => byId("image-file").click());
byId("image-file").addEventListener("change", event => void selectFile(event.target.files[0]));
byId("clear-image").addEventListener("click", clearSelectedImage);
const dropzone = byId("dropzone");
for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", event => void selectFile(event.dataTransfer?.files[0]));
dropzone.addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    byId("image-file").click();
  }
});
byId("model-select").addEventListener("change", updateModelSelection);
byId("run-inspection").addEventListener("click", () => void runInspection());
byId("box-visibility").addEventListener("change", () => renderWorkspaceBoxes(byId("result-content").querySelector(".result-grid > div")));
byId("class-filter").addEventListener("change", () => renderWorkspaceBoxes(byId("result-content").querySelector(".result-grid > div")));
byId("save-result").addEventListener("click", () => {
  if (!activeWorkspaceImage?.complete) return;
  const record = selectDetectionRecord(state.comparisonRecords);
  if (!record) return;
  const image = record.result.image;
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d");
  context.drawImage(activeWorkspaceImage, 0, 0, image.width, image.height);
  if (byId("box-visibility").value === "show") {
    const rows = detectionRows(record.result.detections, image).filter(row =>
      !byId("class-filter").value || row.name === byId("class-filter").value);
    context.lineWidth = Math.max(2, image.width / 400);
    for (const row of rows) {
      const [x1, y1, x2, y2] = row.coordinates;
      context.strokeStyle = colors[CLASSES.indexOf(row.name)];
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      context.fillStyle = "#132931";
      context.fillRect(x1, Math.max(0, y1 - 24), Math.max(80, context.measureText(detectionLabel(row)).width + 8), 24);
      context.fillStyle = colors[CLASSES.indexOf(row.name)];
      context.font = `${Math.max(12, image.width / 80)}px sans-serif`;
      context.fillText(detectionLabel(row), x1 + 4, Math.max(16, y1 - 6));
    }
  }
  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = element("a");
    link.href = url;
    link.download = "pcb-inspection-result.png";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/png");
});
byId("refresh").addEventListener("click", () => void load({overview: true}));
byId("source-refresh").addEventListener("click", async event => {
  const button = event.currentTarget;
  button.disabled = true;
  showNotice("source-feedback", "Checking input sources…");
  try {
    const status = await json("/v1/sources/status", options());
    requirePayload(status?.schema_version === 1 && typeof status.state === "string"
      && Array.isArray(status.sources) && isCount(status.queue_depth) && isCount(status.queue_capacity));
    const summary = status.state === "disabled" ? "Input streaming is disabled." : [
      "Input stream " + status.state,
      status.mode === "mock" ? "Simulated inference" : "",
      ...status.sources.map(source => [source.source_id, source.kind, source.state, source.last_error]
        .filter(Boolean).join(" · ")),
      "Queue " + integer(status.queue_depth) + " / " + integer(status.queue_capacity),
      status.error || "",
    ].filter(Boolean).join(". ") + ".";
    showNotice("source-feedback", summary);
  } catch (error) {
    showNotice("source-feedback", errorMessage(error), true);
  } finally {
    button.disabled = false;
  }
});
for (const name of ["order", "limit"]) {
  byId(name).addEventListener("change", () => {
    state.order = byId("order").value;
    state.limit = Number(byId("limit").value);
    state.offset = 0;
    void load();
  });
}
for (const [id, direction] of [["previous", -1], ["next", 1]]) {
  byId(id).addEventListener("click", () => {
    state.offset = state.page.offset + direction * state.limit;
    void load({focus: true});
  });
}
for (const format of ["csv", "json"]) byId(format).addEventListener("click", () => void exportPage(format));
byId("interval").addEventListener("change", scheduleRefresh);
document.addEventListener("visibilitychange", scheduleRefresh);
byId("close-detail").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", () => {
  state.detailRequest?.abort();
  state.detailRequest = null;
  if (state.imageURL) URL.revokeObjectURL(state.imageURL);
  state.imageURL = null;
  detailContent.replaceChildren();
  state.detailTrigger?.focus();
  state.detailTrigger = null;
  scheduleRefresh();
});
const [from, through] = defaultDateRange();
form.elements.from.value = from;
form.elements.through.value = through;
byId("capture-time").value = new Date(Date.now()).toISOString().slice(0, 16);
applyFilters();
