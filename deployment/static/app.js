const form = document.getElementById("search-form");
const input = document.getElementById("search-input");
const limitSelect = document.getElementById("result-limit");
const results = document.getElementById("results");
const template = document.getElementById("card-template");
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxClose = document.getElementById("lightbox-close");
const lightboxInfoToggle = document.getElementById("lightbox-info-toggle");
const lightboxDelete = document.getElementById("lightbox-delete");
const lightboxMeta = document.getElementById("lightbox-meta");
const lightboxFilename = document.getElementById("lightbox-filename");
const lightboxPath = document.getElementById("lightbox-path");
const lightboxTimeLabel = document.getElementById("lightbox-time-label");
const lightboxTime = document.getElementById("lightbox-time");

let activeController = null;
let activeMetadataController = null;
let activeDeleteUrl = "";
let activeFileName = "";
const DEFAULT_LIMIT = 25;

const isSearchableQuery = (text) => {
    if (text.length >= 2) {
        return true;
    }
    return /[\u4e00-\u9fff]/u.test(text);
};

const closeLightbox = () => {
    lightbox.hidden = true;
    lightboxImage.removeAttribute("src");
    lightboxImage.removeAttribute("alt");
    lightboxMeta.hidden = true;
    lightboxInfoToggle.setAttribute("aria-expanded", "false");
    lightboxFilename.textContent = "";
    lightboxPath.textContent = "";
    lightboxTimeLabel.textContent = "时间";
    lightboxTime.textContent = "";
    activeDeleteUrl = "";
    activeFileName = "";
    lightboxDelete.disabled = false;
    document.body.classList.remove("is-locked");
    if (activeMetadataController) {
        activeMetadataController.abort();
        activeMetadataController = null;
    }
};

const loadMetadata = async (metadataUrl) => {
    if (activeMetadataController) {
        activeMetadataController.abort();
    }
    activeMetadataController = new AbortController();
    const response = await fetch(metadataUrl, { signal: activeMetadataController.signal });
    if (!response.ok) {
        throw new Error("metadata");
    }
    return response.json();
};

const openLightbox = async (fullUrl, altText, fileName, metadataUrl, deleteUrl) => {
    lightboxImage.src = fullUrl;
    lightboxImage.alt = altText;
    lightboxMeta.hidden = true;
    lightboxInfoToggle.setAttribute("aria-expanded", "false");
    lightboxFilename.textContent = fileName || "";
    lightboxPath.textContent = "";
    lightboxTimeLabel.textContent = "时间";
    lightboxTime.textContent = "读取中";
    activeDeleteUrl = deleteUrl || "";
    activeFileName = fileName || "";
    lightboxDelete.disabled = !activeDeleteUrl;
    lightbox.hidden = false;
    document.body.classList.add("is-locked");

    try {
        const metadata = await loadMetadata(metadataUrl);
        lightboxFilename.textContent = metadata.fileName || fileName || "";
        lightboxPath.textContent = metadata.path || "";
        lightboxTimeLabel.textContent = metadata.timeLabel || "时间";
        lightboxTime.textContent = metadata.timeValue || "未知";
    } catch (error) {
        if (error.name !== "AbortError") {
            lightboxPath.textContent = "未知";
            lightboxTimeLabel.textContent = "时间";
            lightboxTime.textContent = "未知";
        }
    }
};

const renderResults = (items) => {
    results.replaceChildren();
    if (!items.length) {
        return;
    }

    const fragment = document.createDocumentFragment();
    for (const item of items) {
        const node = template.content.firstElementChild.cloneNode(true);
        const image = node.querySelector(".result-thumb");
        image.src = item.thumbnailUrl;
        image.alt = item.name;
        node.dataset.fullUrl = item.fullUrl;
        node.dataset.altText = item.name;
        node.dataset.fileName = item.fileName;
        node.dataset.metadataUrl = item.metadataUrl;
        node.dataset.deleteUrl = item.deleteUrl;
        fragment.appendChild(node);
    }
    results.appendChild(fragment);
};

const deleteActiveImage = async () => {
    if (!activeDeleteUrl || lightboxDelete.disabled) {
        return;
    }
    const fileName = activeFileName || lightboxFilename.textContent || "这张图片";
    const confirmed = window.confirm(`永久删除 ${fileName}？删除后会同时从本地相册和当前索引中移除，且无法恢复。`);
    if (!confirmed) {
        return;
    }

    lightboxDelete.disabled = true;
    try {
        const response = await fetch(activeDeleteUrl, { method: "DELETE" });
        if (!response.ok) {
            throw new Error("delete");
        }
        closeLightbox();
        await runSearch(input.value);
    } catch (error) {
        lightboxDelete.disabled = false;
        window.alert("删除失败，请重试。");
    }
};

const normalizedLimit = () => {
    const raw = Number.parseInt(limitSelect.value, 10);
    if (Number.isNaN(raw) || raw < 1) {
        return DEFAULT_LIMIT;
    }
    return Math.min(raw, 100);
};

const runSearch = async (query) => {
    const text = query.trim();
    const params = new URLSearchParams(window.location.search);
    const limit = normalizedLimit();

    if (!isSearchableQuery(text)) {
        params.delete("q");
        params.delete("limit");
        history.replaceState(null, "", `${window.location.pathname}${params.toString() ? `?${params}` : ""}`);
        renderResults([]);
        return;
    }

    params.set("q", text);
    if (limit === DEFAULT_LIMIT) {
        params.delete("limit");
    } else {
        params.set("limit", String(limit));
    }
    history.replaceState(null, "", `${window.location.pathname}?${params}`);

    if (activeController) {
        activeController.abort();
    }
    activeController = new AbortController();

    const response = await fetch(`/api/search?q=${encodeURIComponent(text)}&limit=${limit}`, {
        signal: activeController.signal,
    });
    if (!response.ok) {
        renderResults([]);
        return;
    }

    const payload = await response.json();
    renderResults(payload.results || []);
};

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
        await runSearch(input.value);
    } catch (error) {
        if (error.name !== "AbortError") {
            renderResults([]);
        }
    }
});

limitSelect.addEventListener("change", async () => {
    const query = input.value.trim();
    if (!query) {
        return;
    }
    try {
        await runSearch(query);
    } catch (error) {
        if (error.name !== "AbortError") {
            renderResults([]);
        }
    }
});

results.addEventListener("click", (event) => {
    const card = event.target.closest(".result-card");
    if (!card) {
        return;
    }
    openLightbox(
        card.dataset.fullUrl,
        card.dataset.altText || "",
        card.dataset.fileName || "",
        card.dataset.metadataUrl || "",
        card.dataset.deleteUrl || "",
    );
});

lightboxClose.addEventListener("click", closeLightbox);
lightboxDelete.addEventListener("click", deleteActiveImage);

lightboxInfoToggle.addEventListener("click", () => {
    const nextHidden = !lightboxMeta.hidden;
    lightboxMeta.hidden = nextHidden;
    lightboxInfoToggle.setAttribute("aria-expanded", String(!nextHidden));
});

lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox) {
        closeLightbox();
    }
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !lightbox.hidden) {
        closeLightbox();
    }
});

window.addEventListener("DOMContentLoaded", async () => {
    const params = new URLSearchParams(window.location.search);
    const query = params.get("q");
    const limit = Number.parseInt(params.get("limit") || "", 10);
    if (!Number.isNaN(limit) && [25, 50, 100].includes(limit)) {
        limitSelect.value = String(limit);
    }
    if (!query) {
        return;
    }

    input.value = query;
    try {
        await runSearch(query);
    } catch (error) {
        if (error.name !== "AbortError") {
            renderResults([]);
        }
    }
});
