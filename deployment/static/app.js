const form = document.getElementById("search-form");
const input = document.getElementById("search-input");
const imageInput = document.getElementById("image-input");
const imagePickerButton = document.getElementById("image-picker-button");
const selectionToggle = document.getElementById("selection-toggle");
const selectionBar = document.getElementById("selection-bar");
const selectionCount = document.getElementById("selection-count");
const selectionSelectAll = document.getElementById("selection-select-all");
const selectionClear = document.getElementById("selection-clear");
const selectionDelete = document.getElementById("selection-delete");
const imageQueryChip = document.getElementById("image-query-chip");
const imageQueryPreview = document.getElementById("image-query-preview");
const imageQueryLabel = document.getElementById("image-query-label");
const imageQueryClear = document.getElementById("image-query-clear");
const limitSelect = document.getElementById("result-limit");
const results = document.getElementById("results");
const template = document.getElementById("card-template");
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxClose = document.getElementById("lightbox-close");
const lightboxInfoToggle = document.getElementById("lightbox-info-toggle");
const lightboxDelete = document.getElementById("lightbox-delete");
const lightboxSimilar = document.getElementById("lightbox-similar");
const lightboxMeta = document.getElementById("lightbox-meta");
const lightboxFilename = document.getElementById("lightbox-filename");
const lightboxPath = document.getElementById("lightbox-path");
const lightboxTimeLabel = document.getElementById("lightbox-time-label");
const lightboxTime = document.getElementById("lightbox-time");

let activeController = null;
let activeMetadataController = null;
let activeDeleteUrl = "";
let activeFileName = "";
let activeSimilarUrl = "";
let activeImageQuery = null;
let selectionMode = false;
let selectedPaths = new Set();
const DEFAULT_LIMIT = 25;

const isSearchableQuery = (text) => {
    if (text.length >= 2) {
        return true;
    }
    return /[\u4e00-\u9fff]/u.test(text);
};

const revokeImageQueryPreview = () => {
    if (activeImageQuery?.objectUrl) {
        URL.revokeObjectURL(activeImageQuery.objectUrl);
    }
};

const clearImageQuery = () => {
    revokeImageQueryPreview();
    activeImageQuery = null;
    imageInput.value = "";
    imageQueryPreview.removeAttribute("src");
    imageQueryPreview.removeAttribute("alt");
    imageQueryLabel.textContent = "";
    imageQueryChip.hidden = true;
};

const renderImageQuery = () => {
    if (!activeImageQuery) {
        imageQueryChip.hidden = true;
        imageQueryPreview.removeAttribute("src");
        imageQueryPreview.removeAttribute("alt");
        imageQueryLabel.textContent = "";
        return;
    }

    imageQueryPreview.src = activeImageQuery.previewUrl;
    imageQueryPreview.alt = activeImageQuery.label;
    imageQueryLabel.textContent = activeImageQuery.label;
    imageQueryChip.hidden = false;
};

const setUploadImageQuery = (file) => {
    clearImageQuery();
    const objectUrl = URL.createObjectURL(file);
    activeImageQuery = {
        kind: "upload",
        file,
        label: file.name || "Pasted image",
        previewUrl: objectUrl,
        objectUrl,
    };
    input.value = "";
    renderImageQuery();
};

const setSimilarImageQuery = ({ previewUrl, similarUrl, fileName }) => {
    clearImageQuery();
    activeImageQuery = {
        kind: "gallery",
        similarUrl,
        label: fileName || "Similar image",
        previewUrl,
    };
    input.value = "";
    renderImageQuery();
};

const updateUrlForText = (text, limit) => {
    const params = new URLSearchParams(window.location.search);
    if (!text) {
        params.delete("q");
    } else {
        params.set("q", text);
    }
    if (limit === DEFAULT_LIMIT) {
        params.delete("limit");
    } else {
        params.set("limit", String(limit));
    }
    history.replaceState(null, "", `${window.location.pathname}${params.toString() ? `?${params}` : ""}`);
};

const updateUrlForImage = (limit) => {
    const params = new URLSearchParams(window.location.search);
    params.delete("q");
    if (limit === DEFAULT_LIMIT) {
        params.delete("limit");
    } else {
        params.set("limit", String(limit));
    }
    history.replaceState(null, "", `${window.location.pathname}${params.toString() ? `?${params}` : ""}`);
};

const updateSelectionBar = () => {
    const resultCards = [...results.querySelectorAll(".result-card")];
    const selectableCount = resultCards.length;
    const selectedCount = selectedPaths.size;
    const allSelected = selectableCount > 0 && selectedCount === selectableCount;
    selectionBar.hidden = !selectionMode;
    selectionToggle.classList.toggle("is-active", selectionMode);
    selectionCount.textContent = `${selectedCount} selected`;
    selectionDelete.disabled = selectedCount === 0;
    selectionSelectAll.disabled = selectableCount === 0;
    selectionSelectAll.classList.toggle("is-active", allSelected);
    selectionSelectAll.setAttribute("aria-label", allSelected ? "Clear selected results" : "Select all results");
    selectionSelectAll.title = allSelected ? "Clear selected results" : "Select all results";
};

const exitSelectionMode = () => {
    selectionMode = false;
    selectedPaths = new Set();
    updateSelectionBar();
    renderSelectionState();
};

const renderSelectionState = () => {
    for (const card of results.querySelectorAll(".result-card")) {
        const isSelected = selectedPaths.has(card.dataset.relativePath || "");
        card.classList.toggle("is-selectable", selectionMode);
        card.classList.toggle("is-selected", isSelected);
    }
};

const toggleSelectionMode = () => {
    selectionMode = !selectionMode;
    if (!selectionMode) {
        selectedPaths = new Set();
    }
    updateSelectionBar();
    renderSelectionState();
};

const toggleSelectAll = () => {
    const resultCards = [...results.querySelectorAll(".result-card")];
    if (!resultCards.length) {
        return;
    }
    const allPaths = resultCards
        .map((card) => card.dataset.relativePath || "")
        .filter((path) => path);
    const allSelected = allPaths.length > 0 && allPaths.every((path) => selectedPaths.has(path));
    selectedPaths = allSelected ? new Set() : new Set(allPaths);
    updateSelectionBar();
    renderSelectionState();
};

const closeLightbox = () => {
    lightbox.hidden = true;
    lightboxImage.removeAttribute("src");
    lightboxImage.removeAttribute("alt");
    lightboxMeta.hidden = true;
    lightboxInfoToggle.setAttribute("aria-expanded", "false");
    lightboxFilename.textContent = "";
    lightboxPath.textContent = "";
    lightboxTimeLabel.textContent = "Time";
    lightboxTime.textContent = "";
    activeDeleteUrl = "";
    activeFileName = "";
    activeSimilarUrl = "";
    lightboxDelete.disabled = false;
    lightboxSimilar.disabled = true;
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

const openLightbox = async ({ fullUrl, altText, fileName, metadataUrl, deleteUrl, similarUrl }) => {
    lightboxImage.src = fullUrl;
    lightboxImage.alt = altText;
    lightboxMeta.hidden = true;
    lightboxInfoToggle.setAttribute("aria-expanded", "false");
    lightboxFilename.textContent = fileName || "";
    lightboxPath.textContent = "";
    lightboxTimeLabel.textContent = "Time";
    lightboxTime.textContent = "Loading";
    activeDeleteUrl = deleteUrl || "";
    activeFileName = fileName || "";
    activeSimilarUrl = similarUrl || "";
    lightboxDelete.disabled = !activeDeleteUrl;
    lightboxSimilar.disabled = !activeSimilarUrl;
    lightbox.hidden = false;
    document.body.classList.add("is-locked");

    try {
        const metadata = await loadMetadata(metadataUrl);
        lightboxFilename.textContent = metadata.fileName || fileName || "";
        lightboxPath.textContent = metadata.path || "";
        lightboxTimeLabel.textContent = metadata.timeLabel || "Time";
        lightboxTime.textContent = metadata.timeValue || "Unknown";
    } catch (error) {
        if (error.name !== "AbortError") {
            lightboxPath.textContent = "Unknown";
            lightboxTimeLabel.textContent = "Time";
            lightboxTime.textContent = "Unknown";
        }
    }
};

const renderResults = (items) => {
    results.replaceChildren();
    if (!items.length) {
        if (selectionMode) {
            selectedPaths = new Set();
            updateSelectionBar();
        }
        return;
    }

    const availablePaths = new Set(items.map((item) => item.relativePath));
    selectedPaths = new Set([...selectedPaths].filter((path) => availablePaths.has(path)));

    const fragment = document.createDocumentFragment();
    for (const item of items) {
        const node = template.content.firstElementChild.cloneNode(true);
        const image = node.querySelector(".result-thumb");
        image.src = item.thumbnailUrl;
        image.alt = item.name;
        node.dataset.relativePath = item.relativePath;
        node.dataset.fullUrl = item.fullUrl;
        node.dataset.altText = item.name;
        node.dataset.fileName = item.fileName;
        node.dataset.metadataUrl = item.metadataUrl;
        node.dataset.deleteUrl = item.deleteUrl;
        node.dataset.similarUrl = item.similarUrl;
        fragment.appendChild(node);
    }
    results.appendChild(fragment);
    updateSelectionBar();
    renderSelectionState();
};

const deleteActiveImage = async () => {
    if (!activeDeleteUrl || lightboxDelete.disabled) {
        return;
    }
    const fileName = activeFileName || lightboxFilename.textContent || "this image";
    const confirmed = window.confirm(
        `Delete ${fileName} permanently?\n\nThis removes it from your local gallery and the current index. This action cannot be undone.`,
    );
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
        await runCurrentSearch();
    } catch (error) {
        lightboxDelete.disabled = false;
        window.alert("Delete failed. Please try again.");
    }
};

const deleteSelectedImages = async () => {
    if (selectedPaths.size === 0 || selectionDelete.disabled) {
        return;
    }
    const confirmed = window.confirm(
        `Delete ${selectedPaths.size} selected images permanently?\n\nThis removes them from your local gallery and the current index. This action cannot be undone.`,
    );
    if (!confirmed) {
        return;
    }

    selectionDelete.disabled = true;
    try {
        const response = await fetch("/api/images/batch-delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paths: [...selectedPaths] }),
        });
        if (!response.ok) {
            throw new Error("batch-delete");
        }
        selectedPaths = new Set();
        updateSelectionBar();
        await runCurrentSearch();
    } catch (error) {
        selectionDelete.disabled = false;
        window.alert("Batch delete failed. Please try again.");
    }
};

const normalizedLimit = () => {
    const raw = Number.parseInt(limitSelect.value, 10);
    if (Number.isNaN(raw) || raw < 1) {
        return DEFAULT_LIMIT;
    }
    return Math.min(raw, 100);
};

const fetchResults = async (url, options = {}) => {
    if (activeController) {
        activeController.abort();
    }
    activeController = new AbortController();
    const response = await fetch(url, { ...options, signal: activeController.signal });
    if (!response.ok) {
        renderResults([]);
        return;
    }
    const payload = await response.json();
    renderResults(payload.results || []);
};

const runTextSearch = async (query) => {
    const text = query.trim();
    const limit = normalizedLimit();
    if (!isSearchableQuery(text)) {
        updateUrlForText("", DEFAULT_LIMIT);
        renderResults([]);
        return;
    }

    updateUrlForText(text, limit);
    await fetchResults(`/api/search?q=${encodeURIComponent(text)}&limit=${limit}`);
};

const runImageSearch = async () => {
    if (!activeImageQuery) {
        renderResults([]);
        return;
    }

    const limit = normalizedLimit();
    updateUrlForImage(limit);

    if (activeImageQuery.kind === "gallery") {
        await fetchResults(`${activeImageQuery.similarUrl}?limit=${limit}`);
        return;
    }

    const formData = new FormData();
    formData.append("image", activeImageQuery.file, activeImageQuery.file.name || "query-image");
    await fetchResults(`/api/search/image?limit=${limit}`, { method: "POST", body: formData });
};

const runCurrentSearch = async () => {
    if (activeImageQuery) {
        await runImageSearch();
        return;
    }
    await runTextSearch(input.value);
};

const applyUploadImageFile = async (file) => {
    if (!file) {
        return;
    }
    setUploadImageQuery(file);
    try {
        await runImageSearch();
    } catch (error) {
        if (error.name !== "AbortError") {
            renderResults([]);
        }
    }
};

const maybeExtractPastedImage = (event) => {
    const items = event.clipboardData?.items || [];
    for (const item of items) {
        if (item.kind === "file" && item.type.startsWith("image/")) {
            return item.getAsFile();
        }
    }
    return null;
};

const handlePasteImage = async (event) => {
    const file = maybeExtractPastedImage(event);
    if (!file) {
        return;
    }
    event.preventDefault();
    event.stopPropagation();
    await applyUploadImageFile(file);
};

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
        await runCurrentSearch();
    } catch (error) {
        if (error.name !== "AbortError") {
            renderResults([]);
        }
    }
});

input.addEventListener("paste", handlePasteImage);
form.addEventListener("paste", handlePasteImage);
document.addEventListener("paste", handlePasteImage, true);

imagePickerButton.addEventListener("click", () => {
    imageInput.click();
});

selectionToggle.addEventListener("click", () => {
    toggleSelectionMode();
});

selectionSelectAll.addEventListener("click", () => {
    toggleSelectAll();
});

selectionClear.addEventListener("click", () => {
    exitSelectionMode();
});

selectionDelete.addEventListener("click", async () => {
    await deleteSelectedImages();
});

imageInput.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    await applyUploadImageFile(file);
});

imageQueryClear.addEventListener("click", async () => {
    clearImageQuery();
    try {
        await runCurrentSearch();
    } catch (error) {
        if (error.name !== "AbortError") {
            renderResults([]);
        }
    }
    input.focus();
});

limitSelect.addEventListener("change", async () => {
    if (!input.value.trim() && !activeImageQuery) {
        return;
    }
    try {
        await runCurrentSearch();
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

    if (selectionMode) {
        const relativePath = card.dataset.relativePath || "";
        if (!relativePath) {
            return;
        }
        if (selectedPaths.has(relativePath)) {
            selectedPaths.delete(relativePath);
        } else {
            selectedPaths.add(relativePath);
        }
        updateSelectionBar();
        renderSelectionState();
        return;
    }

    openLightbox({
        fullUrl: card.dataset.fullUrl,
        altText: card.dataset.altText || "",
        fileName: card.dataset.fileName || "",
        metadataUrl: card.dataset.metadataUrl || "",
        deleteUrl: card.dataset.deleteUrl || "",
        similarUrl: card.dataset.similarUrl || "",
    });
});

lightboxClose.addEventListener("click", closeLightbox);
lightboxDelete.addEventListener("click", deleteActiveImage);

lightboxSimilar.addEventListener("click", async () => {
    if (!activeSimilarUrl || lightboxSimilar.disabled) {
        return;
    }
    setSimilarImageQuery({
        previewUrl: lightboxImage.src,
        similarUrl: activeSimilarUrl,
        fileName: activeFileName,
    });
    closeLightbox();
    try {
        await runImageSearch();
    } catch (error) {
        if (error.name !== "AbortError") {
            renderResults([]);
        }
    }
});

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
        return;
    }
    if (event.key === "Escape" && selectionMode) {
        exitSelectionMode();
    }
});

window.addEventListener("beforeunload", () => {
    revokeImageQueryPreview();
});

window.addEventListener("DOMContentLoaded", async () => {
    const params = new URLSearchParams(window.location.search);
    const query = params.get("q");
    const limit = Number.parseInt(params.get("limit") || "", 10);
    if (!Number.isNaN(limit) && [25, 50, 100].includes(limit)) {
        limitSelect.value = String(limit);
    }
    updateSelectionBar();
    if (!query) {
        return;
    }

    input.value = query;
    try {
        await runTextSearch(query);
    } catch (error) {
        if (error.name !== "AbortError") {
            renderResults([]);
        }
    }
});
