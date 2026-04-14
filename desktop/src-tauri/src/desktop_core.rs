use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use base64::Engine;
use chrono::{DateTime, Local};
use exif::{In, Reader as ExifReader, Tag};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageFormat, ImageReader};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager, State};
use tempfile::{Builder as TempfileBuilder, NamedTempFile};
use walkdir::WalkDir;

use crate::{ensure_sidecar, runtime_ready_for_dir, SidecarState};

const INDEX_PROGRESS_EVENT: &str = "semanticgallery://folder-index-progress";
const REFRESH_FAILURE_EVENT: &str = "semanticgallery://folder-refresh-failed";
const SUPPORTED_GALLERY_SUFFIXES: &[&str] = &[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic", ".heif"];
const THUMBNAIL_SIZE: u32 = 512;
const VISIBLE_THUMBNAIL_PREWARM_COUNT: usize = 5;
const MAX_IMAGE_UPLOAD_BYTES: usize = 20 * 1024 * 1024;
const SCAN_PROGRESS_EMIT_INTERVAL_FILES: usize = 128;
const SCAN_PROGRESS_EMIT_INTERVAL_MS: u64 = 150;

#[derive(Clone, Default)]
pub struct DesktopCoreState {
    pub(crate) inner: Arc<Mutex<DesktopCore>>,
}

#[derive(Default)]
pub(crate) struct DesktopCore {
    initialized: bool,
    paths: Option<CorePaths>,
    active_folder: Option<PathBuf>,
    active_encoder_signature: String,
    last_task_message: String,
    active_scan_signature: String,
    search_view: ActiveSearchView,
    index_progress: IndexProgressSnapshot,
    index_started_at: Option<Instant>,
    index_started_at_ms: Option<u64>,
}

#[derive(Clone)]
struct CorePaths {
    app_data_dir: PathBuf,
    index_db_path: PathBuf,
    thumbnails_dir: PathBuf,
    workspace_state_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PersistedWorkspaceState {
    active_folder: Option<String>,
    active_encoder_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStatusPayload {
    setup_status: String,
    active_folder: Option<String>,
    active_encoder_signature: String,
    last_task_message: String,
    indexed_image_count: usize,
    indexing: IndexProgressSnapshot,
    stage2: Stage2Snapshot,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RefreshPayload {
    refreshed: bool,
    skipped: bool,
    #[serde(flatten)]
    runtime: RuntimeStatusPayload,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchPayload {
    query: String,
    results: Vec<SearchResultPayload>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResultPayload {
    path: String,
    relative_path: String,
    file_name: String,
    name: String,
    thumbnail_path: String,
    full_path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MetadataPayload {
    path: String,
    relative_path: String,
    file_name: String,
    byte_size: u64,
    mtime_ns: i64,
    width: u32,
    height: u32,
    time_label: String,
    time_value: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeletePayload {
    deleted: bool,
    file_name: String,
    path: String,
    relative_path: String,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchDeletePayload {
    deleted: Vec<DeletedImagePayload>,
    missing: Vec<String>,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeletedImagePayload {
    file_name: String,
    path: String,
    relative_path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexProgressSnapshot {
    status: String,
    phase: String,
    current: usize,
    total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    started_at_ms: Option<u64>,
    elapsed_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    remaining_seconds: Option<u64>,
    embedded_count: usize,
    reused_count: usize,
    message: String,
}

impl Default for IndexProgressSnapshot {
    fn default() -> Self {
        Self {
            status: "idle".into(),
            phase: "idle".into(),
            current: 0,
            total: 0,
            started_at_ms: None,
            elapsed_seconds: 0,
            remaining_seconds: None,
            embedded_count: 0,
            reused_count: 0,
            message: "No indexing task is running.".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Stage2Snapshot {
    status: String,
    phase: String,
    current: usize,
    total: usize,
    phase_current: usize,
    phase_total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    started_at_ms: Option<u64>,
    elapsed_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    remaining_seconds: Option<u64>,
    message: String,
}

#[derive(Debug, Clone)]
struct ActiveSearchView {
    paths: Vec<String>,
    vectors: Vec<Vec<f32>>,
    normalized_vectors: Vec<Vec<f32>>,
    index_by_path: HashMap<String, usize>,
}

impl Default for ActiveSearchView {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            vectors: Vec::new(),
            normalized_vectors: Vec::new(),
            index_by_path: HashMap::new(),
        }
    }
}

impl ActiveSearchView {
    fn from_rows(rows: Vec<(String, Vec<f32>)>) -> Self {
        let mut paths = Vec::with_capacity(rows.len());
        let mut vectors = Vec::with_capacity(rows.len());
        let mut normalized_vectors = Vec::with_capacity(rows.len());
        let mut index_by_path = HashMap::new();

        for (index, (path, vector)) in rows.into_iter().enumerate() {
            let normalized = normalize_vector(&vector).unwrap_or_else(|| vec![0.0; vector.len()]);
            index_by_path.insert(path.clone(), index);
            paths.push(path);
            vectors.push(vector);
            normalized_vectors.push(normalized);
        }

        Self {
            paths,
            vectors,
            normalized_vectors,
            index_by_path,
        }
    }

    fn search(&self, query_vector: &[f32], limit: usize, exclude_path: Option<&str>) -> Vec<String> {
        if self.paths.is_empty() || limit == 0 {
            return Vec::new();
        }
        let Some(normalized_query) = normalize_vector(query_vector) else {
            return Vec::new();
        };

        let mut scored = self
            .normalized_vectors
            .iter()
            .enumerate()
            .filter_map(|(index, row)| {
                let path = self.paths.get(index)?;
                if exclude_path.is_some_and(|excluded| excluded == path) {
                    return None;
                }
                Some((index, dot_product(row, &normalized_query)))
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
        });
        scored
            .into_iter()
            .take(limit)
            .filter_map(|(index, _)| self.paths.get(index).cloned())
            .collect()
    }

    fn search_similar(&self, image_path: &str, limit: usize) -> Vec<String> {
        let Some(index) = self.index_by_path.get(image_path).copied() else {
            return Vec::new();
        };
        let Some(query_vector) = self.vectors.get(index) else {
            return Vec::new();
        };
        self.search(query_vector, limit, Some(image_path))
    }

    fn without_paths(&self, removed_paths: &[String]) -> Self {
        if removed_paths.is_empty() {
            return self.clone();
        }
        let removed = removed_paths.iter().collect::<HashSet<_>>();
        let rows = self
            .paths
            .iter()
            .zip(self.vectors.iter())
            .filter(|(path, _)| !removed.contains(path))
            .map(|(path, vector)| (path.clone(), vector.clone()))
            .collect::<Vec<_>>();
        Self::from_rows(rows)
    }
}

#[derive(Debug, Clone)]
struct ScanEntry {
    path: PathBuf,
    relative_path: String,
    byte_size: u64,
    mtime_ns: i64,
}

#[derive(Debug, Clone)]
struct FolderScanState {
    file_count: usize,
    total_bytes: u64,
    scan_signature: String,
    rows: Vec<ScanEntry>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct FolderScanProgress {
    files_checked: usize,
    supported_images: usize,
}

#[derive(Debug, Clone)]
struct KnownPathRow {
    absolute_path: String,
    content_hash: String,
    byte_size: u64,
    mtime_ns: i64,
    is_present: bool,
}

#[derive(Debug, Clone)]
struct FolderStateRow {
    active_encoder_signature: String,
    scan_signature: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EncodeVectorPayload {
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Stage2RunResponse {
    active_encoder_signature: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectFolderArgs {
    folder_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefreshFolderArgs {
    lightweight: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchTextArgs {
    query_text: String,
    limit: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchUploadedImageArgs {
    image_base64: String,
    filename: Option<String>,
    limit: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchSimilarArgs {
    relative_path: String,
    limit: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetadataArgs {
    relative_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteImageArgs {
    relative_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteImagesArgs {
    paths: Vec<String>,
}

fn now_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn is_supported_image(path: &Path) -> bool {
    let suffix = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!(".{}", extension.to_ascii_lowercase()));
    suffix
        .as_deref()
        .is_some_and(|candidate| SUPPORTED_GALLERY_SUFFIXES.contains(&candidate))
}

fn normalize_vector(vector: &[f32]) -> Option<Vec<f32>> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 {
        return None;
    }
    Some(vector.iter().map(|value| value / norm).collect())
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| a * b)
        .sum()
}

fn is_searchable_query(query_text: &str) -> bool {
    let query = query_text.trim();
    if query.is_empty() {
        return false;
    }
    if query.chars().count() >= 2 {
        return true;
    }
    query.chars().any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character))
}

fn build_stage2_snapshot(signature: &str) -> Stage2Snapshot {
    if signature != "stage1" {
        return Stage2Snapshot {
            status: "ready".into(),
            phase: "finish".into(),
            current: 1,
            total: 1,
            phase_current: 1,
            phase_total: 1,
            started_at_ms: None,
            elapsed_seconds: 0,
            remaining_seconds: Some(0),
            message: "Stage 2 adaptation is ready.".into(),
        };
    }

    Stage2Snapshot {
        status: "idle".into(),
        phase: "idle".into(),
        current: 0,
        total: 0,
        phase_current: 0,
        phase_total: 0,
        started_at_ms: None,
        elapsed_seconds: 0,
        remaining_seconds: None,
        message: "Stage 2 adaptation is idle.".into(),
    }
}

impl DesktopCore {
    fn ensure_initialized(&mut self, app: &AppHandle) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| format!("Failed to resolve the app support directory: {error}"))?;
        fs::create_dir_all(&app_data_dir)
            .map_err(|error| format!("Failed to create {}: {error}", app_data_dir.display()))?;

        let paths = CorePaths {
            app_data_dir: app_data_dir.clone(),
            index_db_path: app_data_dir.join("index.sqlite3"),
            thumbnails_dir: app_data_dir.join("thumbs"),
            workspace_state_path: app_data_dir.join("workspace-state.json"),
        };
        fs::create_dir_all(&paths.thumbnails_dir)
            .map_err(|error| format!("Failed to create {}: {error}", paths.thumbnails_dir.display()))?;

        let connection = open_connection(&paths.index_db_path)?;
        migrate(&connection)?;

        let persisted = load_workspace_state(&paths.workspace_state_path)?;
        self.paths = Some(paths.clone());
        self.active_folder = persisted
            .active_folder
            .as_deref()
            .map(PathBuf::from)
            .filter(|path| path.is_dir());

        self.active_encoder_signature = persisted
            .active_encoder_signature
            .unwrap_or_else(|| "stage1".into());

        if let Some(folder) = self.active_folder.clone() {
            if let Some(row) = load_folder_state(&connection, &folder)? {
                self.active_encoder_signature = row.active_encoder_signature;
                self.active_scan_signature = row.scan_signature;
            } else {
                self.active_encoder_signature = "stage1".into();
                self.active_scan_signature.clear();
            }
            self.search_view = load_search_view(&connection, &folder, &self.active_encoder_signature)?;
            self.last_task_message = if self.search_view.paths.is_empty() {
                "Choose a folder to build the local index.".into()
            } else {
                "The active folder index is ready.".into()
            };
            self.index_progress = ready_index_progress(self.search_view.paths.len());
            persist_workspace_state(
                &paths.workspace_state_path,
                &PersistedWorkspaceState {
                    active_folder: Some(folder.as_posix()),
                    active_encoder_signature: Some(self.active_encoder_signature.clone()),
                },
            )?;
        } else {
            self.last_task_message = if runtime_ready_for_dir(&paths.app_data_dir) {
                "".into()
            } else {
                "Install the local runtime to prepare Python, search dependencies, and the local encoder."
                    .into()
            };
            self.index_progress = IndexProgressSnapshot::default();
            self.search_view = ActiveSearchView::default();
            self.active_scan_signature.clear();
            persist_workspace_state(&paths.workspace_state_path, &PersistedWorkspaceState::default())?;
        }

        self.initialized = true;
        Ok(())
    }

    fn runtime_status(&mut self, app: &AppHandle) -> Result<RuntimeStatusPayload, String> {
        self.ensure_initialized(app)?;
        Ok(self.snapshot())
    }

    fn select_folder(&mut self, app: &AppHandle, base_url: &str, folder_path: &str) -> Result<RefreshPayload, String> {
        self.ensure_initialized(app)?;
        let resolved = PathBuf::from(folder_path)
            .expand_home()
            .map_err(|error| format!("Failed to resolve the selected folder: {error}"))?
            .canonicalize()
            .map_err(|_| "The selected folder is not available.".to_string())?;
        if !resolved.is_dir() {
            return Err("The selected folder is not available.".into());
        }

        let previous_folder = self.active_folder.clone();
        let previous_signature = self.active_encoder_signature.clone();
        let previous_message = self.last_task_message.clone();
        let previous_scan_signature = self.active_scan_signature.clone();
        let previous_view = self.search_view.clone();
        let previous_index_progress = self.index_progress.clone();

        let connection = self.connection()?;
        self.active_encoder_signature = load_folder_state(&connection, &resolved)?
            .map(|row| row.active_encoder_signature)
            .unwrap_or_else(|| "stage1".into());
        self.active_folder = Some(resolved.clone());
        self.persist_workspace_state()?;

        match self.refresh_folder(app, base_url, false) {
            Ok(payload) => Ok(payload),
            Err(error) => {
                self.active_folder = previous_folder;
                self.active_encoder_signature = previous_signature;
                self.last_task_message = previous_message;
                self.active_scan_signature = previous_scan_signature;
                self.search_view = previous_view;
                self.index_progress = previous_index_progress;
                self.persist_workspace_state()?;
                Err(format!("Failed to index the selected folder: {error}"))
            }
        }
    }

    fn refresh_folder(
        &mut self,
        app: &AppHandle,
        base_url: &str,
        lightweight: bool,
    ) -> Result<RefreshPayload, String> {
        self.ensure_initialized(app)?;
        let folder = self
            .active_folder
            .clone()
            .ok_or_else(|| "Choose a folder before using the desktop app.".to_string())?;

        self.publish_index_progress(app, scan_progress_snapshot(0, 0));

        let mut report_scan_progress = |progress: FolderScanProgress| {
            self.publish_index_progress(app, scan_progress_snapshot(progress.files_checked, progress.supported_images));
        };
        let scan_state = scan_folder(&folder, &mut report_scan_progress)?;
        if lightweight && scan_state.scan_signature == self.active_scan_signature {
            self.last_task_message = "The active folder is already up to date.".into();
            return Ok(RefreshPayload {
                refreshed: false,
                skipped: true,
                runtime: self.snapshot(),
            });
        }

        self.publish_index_progress(
            app,
            IndexProgressSnapshot {
                status: "running".into(),
                phase: "start".into(),
                current: 0,
                total: scan_state.file_count,
                started_at_ms: None,
                elapsed_seconds: 0,
                remaining_seconds: None,
                embedded_count: 0,
                reused_count: 0,
                message: format!(
                    "Found {} supported images. Checking which items need new embeddings.",
                    scan_state.file_count
                ),
            },
        );

        let result: Result<RefreshPayload, String> = (|| {
            let mut connection = self.connection()?;
            let transaction = connection
                .transaction()
                .map_err(|error| format!("Failed to open the local index database: {error}"))?;
            let existing_rows = load_known_paths(&transaction, &folder)?;
            let existing_by_path = existing_rows
                .iter()
                .cloned()
                .map(|row| (row.absolute_path.clone(), row))
                .collect::<HashMap<_, _>>();
            let mut seen_paths = HashSet::new();
            let mut signature_rows = Vec::with_capacity(scan_state.rows.len());
            let mut pending_entries = Vec::new();
            let mut embedded_count = 0usize;
            let mut reused_count = 0usize;

            for entry in &scan_state.rows {
                let absolute_path = entry.path.as_posix();
                seen_paths.insert(absolute_path.clone());
                signature_rows.push((entry.relative_path.clone(), entry.byte_size, entry.mtime_ns));
                let unchanged = existing_by_path.get(&absolute_path).is_some_and(|row| {
                    row.is_present && row.byte_size == entry.byte_size && row.mtime_ns == entry.mtime_ns
                });
                if unchanged
                    && existing_by_path
                        .get(&absolute_path)
                        .and_then(|row| embedding_row_exists(&transaction, &row.content_hash, &self.active_encoder_signature).ok())
                        == Some(true)
                {
                    continue;
                }
                pending_entries.push(entry.clone());
            }

            let ready_count = scan_state.file_count.saturating_sub(pending_entries.len());
            for (index, entry) in pending_entries.iter().enumerate() {
                let absolute_path = entry.path.as_posix();
                let existing_row = existing_by_path.get(&absolute_path);
                let unchanged = existing_row.is_some_and(|row| {
                    row.is_present && row.byte_size == entry.byte_size && row.mtime_ns == entry.mtime_ns
                });
                let content_hash = if unchanged {
                    existing_row
                        .map(|row| row.content_hash.clone())
                        .ok_or_else(|| "The local index is inconsistent.".to_string())?
                } else {
                    sha256_file(&entry.path)?
                };

                if !unchanged {
                    upsert_asset(&transaction, &content_hash, entry.byte_size)?;
                }

                if !embedding_row_exists(&transaction, &content_hash, &self.active_encoder_signature)? {
                    let vector = encode_image_path(
                        base_url,
                        &folder,
                        &self.active_encoder_signature,
                        &entry.path,
                    )?;
                    upsert_embedding(&transaction, &content_hash, &self.active_encoder_signature, &vector)?;
                    embedded_count += 1;
                } else {
                    reused_count += 1;
                }

                upsert_path(
                    &transaction,
                    &absolute_path,
                    &folder.as_posix(),
                    &content_hash,
                    entry.byte_size,
                    entry.mtime_ns,
                    true,
                )?;

                let processed = ready_count + index + 1;
                self.publish_index_progress(
                    app,
                    IndexProgressSnapshot {
                        status: "running".into(),
                        phase: "progress".into(),
                        current: processed,
                        total: scan_state.file_count,
                        started_at_ms: None,
                        elapsed_seconds: 0,
                        remaining_seconds: None,
                        embedded_count,
                        reused_count,
                        message: format!(
                            "Indexing {} ({}/{})",
                            entry.path
                                .file_name()
                                .and_then(|value| value.to_str())
                                .unwrap_or("image"),
                            processed,
                            scan_state.file_count
                        ),
                    },
                );
            }

            mark_missing_paths(&transaction, &folder, &seen_paths)?;
            upsert_folder_state(
                &transaction,
                &folder,
                &self.active_encoder_signature,
                signature_rows.len(),
                scan_state.total_bytes,
                &build_scan_signature(&signature_rows),
            )?;
            transaction
                .commit()
                .map_err(|error| format!("Failed to save the local index: {error}"))?;

            let connection = self.connection()?;
            self.search_view = load_search_view(&connection, &folder, &self.active_encoder_signature)?;
            self.active_scan_signature = load_folder_state(&connection, &folder)?
                .map(|row| row.scan_signature)
                .unwrap_or_else(|| scan_state.scan_signature.clone());
            self.last_task_message = "The active folder index is ready.".into();
            self.index_progress = ready_index_progress(self.search_view.paths.len());
            self.publish_index_progress(app, self.index_progress.clone());
            self.persist_workspace_state()?;
            Ok(RefreshPayload {
                refreshed: true,
                skipped: false,
                runtime: self.snapshot(),
            })
        })();

        if let Err(message) = &result {
            self.last_task_message = message.clone();
            self.publish_index_progress(
                app,
                IndexProgressSnapshot {
                    status: "failed".into(),
                    phase: "failed".into(),
                    current: 0,
                    total: scan_state.file_count,
                    started_at_ms: None,
                    elapsed_seconds: 0,
                    remaining_seconds: None,
                    embedded_count: 0,
                    reused_count: 0,
                    message: format!("Failed to refresh the active folder: {message}"),
                },
            );
            let _ = app.emit(
                REFRESH_FAILURE_EVENT,
                serde_json::json!({ "message": format!("Failed to refresh the active folder: {message}") }),
            );
        }

        result
    }

    fn search_text(&mut self, app: &AppHandle, base_url: &str, query_text: &str, limit: usize) -> Result<SearchPayload, String> {
        self.ensure_initialized(app)?;
        let text = query_text.trim().to_string();
        if !is_searchable_query(&text) || limit == 0 || self.search_view.paths.is_empty() {
            return Ok(SearchPayload {
                query: text,
                results: Vec::new(),
            });
        }
        let folder = self
            .active_folder
            .clone()
            .ok_or_else(|| "Choose a folder before using the desktop app.".to_string())?;
        let vector = encode_text(base_url, &folder, &self.active_encoder_signature, &text)?;
        let matches = self.search_view.search(&vector, limit, None);
        self.build_search_payload(&folder, &text, matches)
    }

    fn search_uploaded_image(
        &mut self,
        app: &AppHandle,
        base_url: &str,
        image_base64: &str,
        filename: Option<&str>,
        limit: usize,
    ) -> Result<SearchPayload, String> {
        self.ensure_initialized(app)?;
        if limit == 0 || self.search_view.paths.is_empty() {
            return Ok(SearchPayload {
                query: String::new(),
                results: Vec::new(),
            });
        }
        let folder = self
            .active_folder
            .clone()
            .ok_or_else(|| "Choose a folder before using the desktop app.".to_string())?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(image_base64)
            .map_err(|_| "Unsupported image payload.".to_string())?;
        if bytes.is_empty() {
            return Err("Image is empty.".into());
        }
        if bytes.len() > MAX_IMAGE_UPLOAD_BYTES {
            return Err("Image is too large.".into());
        }
        let temp_file = write_uploaded_image(&bytes, filename)?;
        let vector = encode_image_path(base_url, &folder, &self.active_encoder_signature, temp_file.path())?;
        let matches = self.search_view.search(&vector, limit, None);
        self.build_search_payload(&folder, "", matches)
    }

    fn search_similar(
        &mut self,
        app: &AppHandle,
        base_url: &str,
        relative_path: &str,
        limit: usize,
    ) -> Result<SearchPayload, String> {
        self.ensure_initialized(app)?;
        if limit == 0 || self.search_view.paths.is_empty() {
            return Ok(SearchPayload {
                query: String::new(),
                results: Vec::new(),
            });
        }
        let folder = self
            .active_folder
            .clone()
            .ok_or_else(|| "Choose a folder before using the desktop app.".to_string())?;
        let file_path = resolve_active_file(&folder, relative_path)?;
        let file_key = file_path.as_posix();
        let matches = {
            let direct = self.search_view.search_similar(&file_key, limit);
            if direct.is_empty() {
                let vector = encode_image_path(base_url, &folder, &self.active_encoder_signature, &file_path)?;
                self.search_view.search(&vector, limit, Some(&file_key))
            } else {
                direct
            }
        };
        self.build_search_payload(&folder, "", matches)
    }

    fn metadata(&mut self, app: &AppHandle, relative_path: &str) -> Result<MetadataPayload, String> {
        self.ensure_initialized(app)?;
        let folder = self
            .active_folder
            .clone()
            .ok_or_else(|| "Choose a folder before using the desktop app.".to_string())?;
        let file_path = resolve_active_file(&folder, relative_path)?;
        let stat = file_path
            .metadata()
            .map_err(|_| "Image not found.".to_string())?;
        let (width, height) = image::image_dimensions(&file_path).unwrap_or((0, 0));
        let file_time = stat.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let (time_label, time_value) = read_image_timestamp(&file_path)
            .map(|value| ("Capture time".to_string(), value))
            .unwrap_or_else(|| ("File time".to_string(), format_local_time(file_time)));
        Ok(MetadataPayload {
            path: file_path.as_posix(),
            relative_path: file_path
                .strip_prefix(&folder)
                .unwrap_or(&file_path)
                .as_posix(),
            file_name: file_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("Image")
                .to_string(),
            byte_size: stat.len(),
            mtime_ns: modified_time_ns(&stat),
            width,
            height,
            time_label,
            time_value,
        })
    }

    fn delete_image(&mut self, app: &AppHandle, relative_path: &str) -> Result<DeletePayload, String> {
        self.ensure_initialized(app)?;
        let batch = self.delete_images(app, &[relative_path.to_string()])?;
        let deleted = batch
            .deleted
            .into_iter()
            .next()
            .ok_or_else(|| "Image not found.".to_string())?;
        Ok(DeletePayload {
            deleted: true,
            file_name: deleted.file_name,
            path: deleted.path,
            relative_path: deleted.relative_path,
            message: batch.message,
        })
    }

    fn delete_images(&mut self, app: &AppHandle, paths: &[String]) -> Result<BatchDeletePayload, String> {
        self.ensure_initialized(app)?;
        let folder = self
            .active_folder
            .clone()
            .ok_or_else(|| "Choose a folder before using the desktop app.".to_string())?;
        let mut unique_paths = Vec::new();
        let mut missing = Vec::new();
        let mut seen = HashSet::new();

        for relative_path in paths {
            match resolve_active_file(&folder, relative_path) {
                Ok(file_path) => {
                    let key = file_path.as_posix();
                    if seen.insert(key.clone()) {
                        unique_paths.push(file_path);
                    }
                }
                Err(_) => missing.push(relative_path.clone()),
            }
        }

        for path in &unique_paths {
            trash::delete(path).map_err(|error| format!("Failed to move the selected images to the Trash: {error}"))?;
            delete_thumbnail(self.thumbnail_cache_path(path));
        }

        self.drop_deleted_paths(&folder, &unique_paths)?;
        let count = unique_paths.len();
        let message = if count == 1 {
            "Moved 1 image to the Trash.".to_string()
        } else {
            format!("Moved {count} images to the Trash.")
        };
        self.last_task_message = message.clone();
        let deleted = unique_paths
            .iter()
            .map(|path| DeletedImagePayload {
                file_name: path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("Image")
                    .to_string(),
                path: path.as_posix(),
                relative_path: path
                    .strip_prefix(&folder)
                    .unwrap_or(path)
                    .as_posix(),
            })
            .collect::<Vec<_>>();
        let _ = app.emit(INDEX_PROGRESS_EVENT, serde_json::to_value(&self.index_progress).unwrap_or_default());
        Ok(BatchDeletePayload {
            deleted,
            missing,
            message,
        })
    }

    fn run_stage2(&mut self, app: &AppHandle, base_url: &str) -> Result<RuntimeStatusPayload, String> {
        self.ensure_initialized(app)?;
        let folder = self
            .active_folder
            .clone()
            .ok_or_else(|| "Choose a folder before using the desktop app.".to_string())?;
        let response = post_json::<_, Stage2RunResponse>(base_url, "/api/stage2/run", &serde_json::json!({}))?;
        self.active_encoder_signature = if response.active_encoder_signature.is_empty() {
            "stage1".into()
        } else {
            response.active_encoder_signature
        };
        let connection = self.connection()?;
        self.search_view = load_search_view(&connection, &folder, &self.active_encoder_signature)?;
        self.active_scan_signature = load_folder_state(&connection, &folder)?
            .map(|row| row.scan_signature)
            .unwrap_or_default();
        self.last_task_message = "Stage 2 adaptation is ready.".into();
        self.index_progress = ready_index_progress(self.search_view.paths.len());
        self.persist_workspace_state()?;
        Ok(self.snapshot())
    }

    fn build_search_payload(
        &self,
        folder: &Path,
        query: &str,
        absolute_paths: Vec<String>,
    ) -> Result<SearchPayload, String> {
        let mut results = Vec::with_capacity(absolute_paths.len());
        let mut file_paths = Vec::with_capacity(absolute_paths.len());
        for absolute_path in absolute_paths {
            let file_path = PathBuf::from(&absolute_path);
            let thumbnail_path = self.thumbnail_cache_path(&file_path);
            results.push(SearchResultPayload {
                path: absolute_path.clone(),
                relative_path: file_path
                    .strip_prefix(folder)
                    .unwrap_or(&file_path)
                    .as_posix(),
                file_name: file_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("Image")
                    .to_string(),
                name: file_path
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("Image")
                    .to_string(),
                thumbnail_path: thumbnail_path.as_posix(),
                full_path: file_path.as_posix(),
            });
            file_paths.push(file_path);
        }
        self.prewarm_search_result_thumbnails(file_paths);
        Ok(SearchPayload {
            query: query.to_string(),
            results,
        })
    }

    fn publish_index_progress(&mut self, app: &AppHandle, mut payload: IndexProgressSnapshot) {
        match payload.status.as_str() {
            "running" => {
                if self.index_started_at.is_none() || self.index_started_at_ms.is_none() {
                    self.index_started_at = Some(Instant::now());
                    self.index_started_at_ms = Some(now_epoch_millis());
                }
                let elapsed = self
                    .index_started_at
                    .map(|started_at| started_at.elapsed().as_secs())
                    .unwrap_or(0);
                payload.started_at_ms = self.index_started_at_ms;
                payload.elapsed_seconds = elapsed;
                payload.remaining_seconds = if payload.current > 0 && payload.total >= payload.current {
                    Some(((elapsed as f64 / payload.current as f64) * (payload.total - payload.current) as f64).round() as u64)
                } else {
                    None
                };
            }
            "ready" => {
                payload.started_at_ms = self.index_started_at_ms;
                payload.elapsed_seconds = self
                    .index_started_at
                    .map(|started_at| started_at.elapsed().as_secs())
                    .unwrap_or(0);
                payload.remaining_seconds = Some(0);
                self.index_started_at = None;
                self.index_started_at_ms = None;
            }
            _ => {
                payload.started_at_ms = None;
                payload.elapsed_seconds = 0;
                payload.remaining_seconds = None;
                self.index_started_at = None;
                self.index_started_at_ms = None;
            }
        }

        self.last_task_message = payload.message.clone();
        self.index_progress = payload.clone();
        let _ = app.emit(INDEX_PROGRESS_EVENT, serde_json::to_value(payload).unwrap_or_default());
    }

    fn drop_deleted_paths(&mut self, folder: &Path, deleted_paths: &[PathBuf]) -> Result<(), String> {
        let removed = deleted_paths
            .iter()
            .map(|path| path.as_posix())
            .collect::<Vec<_>>();
        self.search_view = self.search_view.without_paths(&removed);
        let mut connection = self.connection()?;
        let transaction = connection
            .transaction()
            .map_err(|error| format!("Failed to open the local index database: {error}"))?;
        for absolute_path in &removed {
            transaction
                .execute(
                    "UPDATE image_paths SET is_present = 0, last_scanned_at = CURRENT_TIMESTAMP WHERE absolute_path = ?",
                    params![absolute_path],
                )
                .map_err(|error| format!("Failed to update the local index: {error}"))?;
        }
        let present_rows = load_present_path_rows(&transaction, folder)?;
        let signature_rows = present_rows
            .iter()
            .filter_map(|row| {
                PathBuf::from(&row.absolute_path)
                    .strip_prefix(folder)
                    .ok()
                    .map(|relative| (relative.as_posix(), row.byte_size, row.mtime_ns))
            })
            .collect::<Vec<_>>();
        self.active_scan_signature = build_scan_signature(&signature_rows);
        upsert_folder_state(
            &transaction,
            folder,
            &self.active_encoder_signature,
            signature_rows.len(),
            present_rows.iter().map(|row| row.byte_size).sum(),
            &self.active_scan_signature,
        )?;
        transaction
            .commit()
            .map_err(|error| format!("Failed to save the local index: {error}"))?;
        self.index_progress = ready_index_progress(self.search_view.paths.len());
        self.persist_workspace_state()?;
        Ok(())
    }

    fn snapshot(&self) -> RuntimeStatusPayload {
        RuntimeStatusPayload {
            setup_status: "ready".into(),
            active_folder: self.active_folder.as_ref().map(|path| path.as_posix()),
            active_encoder_signature: if self.active_encoder_signature.is_empty() {
                "stage1".into()
            } else {
                self.active_encoder_signature.clone()
            },
            last_task_message: self.last_task_message.clone(),
            indexed_image_count: self.search_view.paths.len(),
            indexing: self.index_progress.clone(),
            stage2: build_stage2_snapshot(&self.active_encoder_signature),
        }
    }

    fn connection(&self) -> Result<Connection, String> {
        let paths = self
            .paths
            .as_ref()
            .ok_or_else(|| "The desktop app has not been initialized.".to_string())?;
        let connection = open_connection(&paths.index_db_path)?;
        migrate(&connection)?;
        Ok(connection)
    }

    fn persist_workspace_state(&self) -> Result<(), String> {
        let paths = self
            .paths
            .as_ref()
            .ok_or_else(|| "The desktop app has not been initialized.".to_string())?;
        persist_workspace_state(
            &paths.workspace_state_path,
            &PersistedWorkspaceState {
                active_folder: self.active_folder.as_ref().map(|path| path.as_posix()),
                active_encoder_signature: Some(if self.active_encoder_signature.is_empty() {
                    "stage1".into()
                } else {
                    self.active_encoder_signature.clone()
                }),
            },
        )
    }

    fn thumbnail_cache_path(&self, file_path: &Path) -> PathBuf {
        let paths = self.paths.as_ref().expect("desktop paths should exist");
        thumbnail_cache_path(&paths.thumbnails_dir, file_path)
    }

    fn prewarm_search_result_thumbnails(&self, file_paths: Vec<PathBuf>) {
        if file_paths.is_empty() {
            return;
        }
        let paths = self.paths.as_ref().expect("desktop paths should exist");
        prewarm_thumbnails(&paths.thumbnails_dir, &file_paths[..file_paths.len().min(VISIBLE_THUMBNAIL_PREWARM_COUNT)]);
        if file_paths.len() > VISIBLE_THUMBNAIL_PREWARM_COUNT {
            let thumbnails_dir = paths.thumbnails_dir.clone();
            let background_paths = file_paths[VISIBLE_THUMBNAIL_PREWARM_COUNT..].to_vec();
            thread::spawn(move || {
                prewarm_thumbnails(&thumbnails_dir, &background_paths);
            });
        }
    }
}

fn ready_index_progress(indexed_count: usize) -> IndexProgressSnapshot {
    IndexProgressSnapshot {
        status: "ready".into(),
        phase: "finish".into(),
        current: indexed_count,
        total: indexed_count,
        started_at_ms: None,
        elapsed_seconds: 0,
        remaining_seconds: Some(0),
        embedded_count: 0,
        reused_count: 0,
        message: "The folder index is ready.".into(),
    }
}

fn scan_progress_snapshot(files_checked: usize, supported_images: usize) -> IndexProgressSnapshot {
    let message = if files_checked == 0 {
        "Scanning the selected folder for index updates.".to_string()
    } else if supported_images == 0 {
        format!("Scanning the selected folder for index updates. Checked {files_checked} files.")
    } else {
        format!(
            "Scanning the selected folder for index updates. Checked {files_checked} files and found {supported_images} supported images."
        )
    };

    IndexProgressSnapshot {
        status: "running".into(),
        phase: "scan".into(),
        current: files_checked,
        total: 0,
        started_at_ms: None,
        elapsed_seconds: 0,
        remaining_seconds: None,
        embedded_count: 0,
        reused_count: 0,
        message,
    }
}

fn open_connection(index_db_path: &Path) -> Result<Connection, String> {
    Connection::open(index_db_path)
        .map_err(|error| format!("Failed to open the local index database: {error}"))
}

fn migrate(connection: &Connection) -> Result<(), String> {
    connection
        .execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS image_assets (
              content_hash TEXT PRIMARY KEY,
              byte_size INTEGER NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS image_embeddings (
              content_hash TEXT NOT NULL,
              encoder_signature TEXT NOT NULL,
              embedding_blob BLOB NOT NULL,
              embedding_dim INTEGER NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (content_hash, encoder_signature)
            );
            CREATE TABLE IF NOT EXISTS image_paths (
              path_id INTEGER PRIMARY KEY AUTOINCREMENT,
              absolute_path TEXT NOT NULL UNIQUE,
              folder_path TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              byte_size INTEGER NOT NULL,
              mtime_ns INTEGER NOT NULL,
              is_present INTEGER NOT NULL,
              last_scanned_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_image_paths_folder_present_order_content
              ON image_paths(folder_path, is_present, absolute_path, content_hash);
            CREATE INDEX IF NOT EXISTS idx_image_embeddings_signature_content
              ON image_embeddings(encoder_signature, content_hash);
            CREATE TABLE IF NOT EXISTS folder_states (
              folder_path TEXT PRIMARY KEY,
              active_encoder_signature TEXT NOT NULL,
              file_count INTEGER NOT NULL DEFAULT 0,
              total_bytes INTEGER NOT NULL DEFAULT 0,
              scan_signature TEXT NOT NULL DEFAULT '',
              last_scanned_at TEXT,
              last_synced_at TEXT
            );
            "#,
        )
        .map_err(|error| format!("Failed to migrate the local index database: {error}"))
}

fn load_workspace_state(path: &Path) -> Result<PersistedWorkspaceState, String> {
    if !path.is_file() {
        return Ok(PersistedWorkspaceState::default());
    }
    let body = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&body)
        .map_err(|error| format!("Failed to read the desktop workspace state: {error}"))
}

fn persist_workspace_state(path: &Path, state: &PersistedWorkspaceState) -> Result<(), String> {
    let body = serde_json::to_string_pretty(state)
        .map_err(|error| format!("Failed to serialize the desktop workspace state: {error}"))?;
    fs::write(path, body).map_err(|error| format!("Failed to write {}: {error}", path.display()))
}

fn load_folder_state(connection: &Connection, folder: &Path) -> Result<Option<FolderStateRow>, String> {
    let mut statement = connection
        .prepare(
            "SELECT active_encoder_signature, scan_signature
             FROM folder_states
             WHERE folder_path = ?",
        )
        .map_err(|error| format!("Failed to read the folder state: {error}"))?;
    statement
        .query_row(params![folder.as_posix()], |row| {
            Ok(FolderStateRow {
                active_encoder_signature: row.get::<_, String>(0)?,
                scan_signature: row.get::<_, String>(1)?,
            })
        })
        .optional()
        .map_err(|error| format!("Failed to read the folder state: {error}"))
}

fn load_search_view(connection: &Connection, folder: &Path, encoder_signature: &str) -> Result<ActiveSearchView, String> {
    let mut statement = connection
        .prepare(
            "SELECT image_paths.absolute_path, image_embeddings.embedding_blob, image_embeddings.embedding_dim
             FROM image_paths
             JOIN image_embeddings
               ON image_paths.content_hash = image_embeddings.content_hash
             WHERE image_paths.folder_path = ?
               AND image_paths.is_present = 1
               AND image_embeddings.encoder_signature = ?
             ORDER BY image_paths.absolute_path",
        )
        .map_err(|error| format!("Failed to load the local search view: {error}"))?;
    let rows = statement
        .query_map(params![folder.as_posix(), encoder_signature], |row| {
            let absolute_path = row.get::<_, String>(0)?;
            let blob = row.get::<_, Vec<u8>>(1)?;
            let embedding_dim = row.get::<_, usize>(2)?;
            Ok((absolute_path, decode_embedding_blob(&blob, embedding_dim)))
        })
        .map_err(|error| format!("Failed to load the local search view: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Failed to load the local search view: {error}"))?;
    Ok(ActiveSearchView::from_rows(rows))
}

fn decode_embedding_blob(blob: &[u8], embedding_dim: usize) -> Vec<f32> {
    blob.chunks_exact(4)
        .take(embedding_dim)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn load_known_paths(connection: &Connection, folder: &Path) -> Result<Vec<KnownPathRow>, String> {
    let mut statement = connection
        .prepare(
            "SELECT absolute_path, content_hash, byte_size, mtime_ns, is_present
             FROM image_paths
             WHERE folder_path = ?
             ORDER BY absolute_path",
        )
        .map_err(|error| format!("Failed to read the local index: {error}"))?;
    let rows = statement
        .query_map(params![folder.as_posix()], |row| {
            Ok(KnownPathRow {
                absolute_path: row.get::<_, String>(0)?,
                content_hash: row.get::<_, String>(1)?,
                byte_size: row.get::<_, u64>(2)?,
                mtime_ns: row.get::<_, i64>(3)?,
                is_present: row.get::<_, i64>(4)? == 1,
            })
        })
        .map_err(|error| format!("Failed to read the local index: {error}"))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Failed to read the local index: {error}"))
}

fn load_present_path_rows(connection: &Connection, folder: &Path) -> Result<Vec<KnownPathRow>, String> {
    load_known_paths(connection, folder).map(|rows| rows.into_iter().filter(|row| row.is_present).collect())
}

fn embedding_row_exists(connection: &Connection, content_hash: &str, encoder_signature: &str) -> Result<bool, String> {
    let mut statement = connection
        .prepare(
            "SELECT 1
             FROM image_embeddings
             WHERE content_hash = ? AND encoder_signature = ?",
        )
        .map_err(|error| format!("Failed to read the local index: {error}"))?;
    statement
        .exists(params![content_hash, encoder_signature])
        .map_err(|error| format!("Failed to read the local index: {error}"))
}

fn upsert_asset(connection: &Connection, content_hash: &str, byte_size: u64) -> Result<(), String> {
    connection
        .execute(
            "INSERT INTO image_assets (content_hash, byte_size)
             VALUES (?, ?)
             ON CONFLICT(content_hash) DO UPDATE SET
               byte_size = excluded.byte_size,
               updated_at = CURRENT_TIMESTAMP",
            params![content_hash, byte_size],
        )
        .map(|_| ())
        .map_err(|error| format!("Failed to update the local index: {error}"))
}

fn upsert_embedding(
    connection: &Connection,
    content_hash: &str,
    encoder_signature: &str,
    embedding: &[f32],
) -> Result<(), String> {
    let mut blob = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    connection
        .execute(
            "INSERT INTO image_embeddings (content_hash, encoder_signature, embedding_blob, embedding_dim)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(content_hash, encoder_signature) DO UPDATE SET
               embedding_blob = excluded.embedding_blob,
               embedding_dim = excluded.embedding_dim",
            params![content_hash, encoder_signature, blob, embedding.len()],
        )
        .map(|_| ())
        .map_err(|error| format!("Failed to update the local index: {error}"))
}

fn upsert_path(
    connection: &Connection,
    absolute_path: &str,
    folder_path: &str,
    content_hash: &str,
    byte_size: u64,
    mtime_ns: i64,
    is_present: bool,
) -> Result<(), String> {
    connection
        .execute(
            "INSERT INTO image_paths (absolute_path, folder_path, content_hash, byte_size, mtime_ns, is_present)
             VALUES (?, ?, ?, ?, ?, ?)
             ON CONFLICT(absolute_path) DO UPDATE SET
               folder_path = excluded.folder_path,
               content_hash = excluded.content_hash,
               byte_size = excluded.byte_size,
               mtime_ns = excluded.mtime_ns,
               is_present = excluded.is_present,
               last_scanned_at = CURRENT_TIMESTAMP",
            params![absolute_path, folder_path, content_hash, byte_size, mtime_ns, if is_present { 1 } else { 0 }],
        )
        .map(|_| ())
        .map_err(|error| format!("Failed to update the local index: {error}"))
}

fn mark_missing_paths(connection: &Connection, folder: &Path, seen_paths: &HashSet<String>) -> Result<(), String> {
    for row in load_known_paths(connection, folder)? {
        if row.is_present && !seen_paths.contains(&row.absolute_path) {
            connection
                .execute(
                    "UPDATE image_paths
                     SET is_present = 0,
                         last_scanned_at = CURRENT_TIMESTAMP
                     WHERE absolute_path = ?",
                    params![row.absolute_path],
                )
                .map_err(|error| format!("Failed to update the local index: {error}"))?;
        }
    }
    Ok(())
}

fn upsert_folder_state(
    connection: &Connection,
    folder: &Path,
    active_encoder_signature: &str,
    file_count: usize,
    total_bytes: u64,
    scan_signature: &str,
) -> Result<(), String> {
    connection
        .execute(
            "INSERT INTO folder_states (
               folder_path,
               active_encoder_signature,
               file_count,
               total_bytes,
               scan_signature,
               last_scanned_at,
               last_synced_at
             )
             VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
             ON CONFLICT(folder_path) DO UPDATE SET
               active_encoder_signature = excluded.active_encoder_signature,
               file_count = excluded.file_count,
               total_bytes = excluded.total_bytes,
               scan_signature = excluded.scan_signature,
               last_scanned_at = CURRENT_TIMESTAMP,
               last_synced_at = CURRENT_TIMESTAMP",
            params![
                folder.as_posix(),
                active_encoder_signature,
                file_count,
                total_bytes,
                scan_signature
            ],
        )
        .map(|_| ())
        .map_err(|error| format!("Failed to update the local index: {error}"))
}

fn scan_folder<F>(folder: &Path, on_progress: &mut F) -> Result<FolderScanState, String>
where
    F: FnMut(FolderScanProgress),
{
    let mut entries = Vec::new();
    let mut files_checked = 0usize;
    let mut supported_images = 0usize;
    let mut last_reported = FolderScanProgress::default();
    let mut last_progress_emit = Instant::now();

    for entry in WalkDir::new(folder).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }

        files_checked += 1;
        let path = entry.path().to_path_buf();
        let relative_path = match path.strip_prefix(folder).ok() {
            Some(value) => value.as_posix(),
            None => {
                maybe_emit_scan_progress(
                    on_progress,
                    &mut last_reported,
                    &mut last_progress_emit,
                    files_checked,
                    supported_images,
                    false,
                );
                continue;
            }
        };

        if relative_path.split('/').any(|part| part.starts_with('.')) || !is_supported_image(&path) {
            maybe_emit_scan_progress(
                on_progress,
                &mut last_reported,
                &mut last_progress_emit,
                files_checked,
                supported_images,
                false,
            );
            continue;
        }

        let metadata = match path.metadata() {
            Ok(value) => value,
            Err(_) => {
                maybe_emit_scan_progress(
                    on_progress,
                    &mut last_reported,
                    &mut last_progress_emit,
                    files_checked,
                    supported_images,
                    false,
                );
                continue;
            }
        };

        supported_images += 1;
        entries.push(ScanEntry {
            path: path.clone(),
            relative_path,
            byte_size: metadata.len(),
            mtime_ns: modified_time_ns(&metadata),
        });

        maybe_emit_scan_progress(
            on_progress,
            &mut last_reported,
            &mut last_progress_emit,
            files_checked,
            supported_images,
            false,
        );
    }

    maybe_emit_scan_progress(
        on_progress,
        &mut last_reported,
        &mut last_progress_emit,
        files_checked,
        supported_images,
        true,
    );

    entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let total_bytes = entries.iter().map(|entry| entry.byte_size).sum();
    let signature_rows = entries
        .iter()
        .map(|entry| (entry.relative_path.clone(), entry.byte_size, entry.mtime_ns))
        .collect::<Vec<_>>();
    Ok(FolderScanState {
        file_count: entries.len(),
        total_bytes,
        scan_signature: build_scan_signature(&signature_rows),
        rows: entries,
    })
}

fn maybe_emit_scan_progress<F>(
    on_progress: &mut F,
    last_reported: &mut FolderScanProgress,
    last_progress_emit: &mut Instant,
    files_checked: usize,
    supported_images: usize,
    force: bool,
) where
    F: FnMut(FolderScanProgress),
{
    let progress = FolderScanProgress {
        files_checked,
        supported_images,
    };

    if progress.files_checked == 0 || progress == *last_reported {
        return;
    }

    let crossed_file_interval =
        progress.files_checked.saturating_sub(last_reported.files_checked) >= SCAN_PROGRESS_EMIT_INTERVAL_FILES;
    let crossed_time_interval =
        last_progress_emit.elapsed() >= Duration::from_millis(SCAN_PROGRESS_EMIT_INTERVAL_MS);
    if !force && !crossed_file_interval && !crossed_time_interval {
        return;
    }

    *last_reported = progress;
    *last_progress_emit = Instant::now();
    on_progress(progress);
}

fn build_scan_signature(rows: &[(String, u64, i64)]) -> String {
    let mut digest = Sha256::new();
    for (relative_path, size, mtime_ns) in rows {
        digest.update(relative_path.as_bytes());
        digest.update(b"\0");
        digest.update(size.to_string().as_bytes());
        digest.update(b"\0");
        digest.update(mtime_ns.to_string().as_bytes());
        digest.update(b"\n");
    }
    format!("{:x}", digest.finalize())
}

fn modified_time_ns(metadata: &fs::Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos() as i64)
        .unwrap_or(0)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut digest = Sha256::new();
    let mut handle = File::open(path).map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = handle
            .read(&mut buffer)
            .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn encode_text(base_url: &str, folder: &Path, encoder_signature: &str, query_text: &str) -> Result<Vec<f32>, String> {
    let payload = serde_json::json!({
        "folderPath": folder.as_posix(),
        "encoderSignature": encoder_signature,
        "queryText": query_text
    });
    post_json::<_, EncodeVectorPayload>(base_url, "/api/encode/text", &payload).map(|response| response.vector)
}

fn encode_image_path(
    base_url: &str,
    folder: &Path,
    encoder_signature: &str,
    image_path: &Path,
) -> Result<Vec<f32>, String> {
    let payload = serde_json::json!({
        "folderPath": folder.as_posix(),
        "encoderSignature": encoder_signature,
        "imagePath": image_path.as_posix()
    });
    post_json::<_, EncodeVectorPayload>(base_url, "/api/encode/image-path", &payload).map(|response| response.vector)
}

fn post_json<TBody: Serialize, TResponse: for<'de> Deserialize<'de>>(
    base_url: &str,
    path: &str,
    body: &TBody,
) -> Result<TResponse, String> {
    let response = match ureq::post(&format!("{base_url}{path}")).send_json(
        serde_json::to_value(body).map_err(|error| format!("Failed to build the local request: {error}"))?,
    ) {
        Ok(response) => response,
        Err(ureq::Error::Status(_, response)) => {
            let detail = response
                .into_json::<serde_json::Value>()
                .ok()
                .and_then(|payload: serde_json::Value| {
                    payload
                        .get("detail")
                        .and_then(|value| value.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "The local runtime returned an error.".into());
            return Err(detail);
        }
        Err(ureq::Error::Transport(error)) => {
            return Err(format!("Failed to reach the local runtime: {error}"));
        }
    };
    response
        .into_json::<TResponse>()
        .map_err(|error| format!("Failed to read the local runtime response: {error}"))
}

fn resolve_active_file(folder: &Path, relative_path: &str) -> Result<PathBuf, String> {
    let candidate = folder.join(relative_path);
    let resolved = candidate
        .canonicalize()
        .map_err(|_| "Image not found.".to_string())?;
    if !resolved.starts_with(folder) || !resolved.is_file() {
        return Err("Image not found.".into());
    }
    Ok(resolved)
}

fn thumbnail_cache_path(thumbnails_dir: &Path, file_path: &Path) -> PathBuf {
    let metadata = file_path.metadata();
    let cache_key = if let Ok(metadata) = metadata {
        format!("{}:{}:{}", file_path.as_posix(), modified_time_ns(&metadata), metadata.len())
    } else {
        file_path.as_posix()
    };
    let mut digest = Sha1::new();
    digest.update(cache_key.as_bytes());
    thumbnails_dir.join(format!("{:x}.png", digest.finalize()))
}

fn prewarm_thumbnails(thumbnails_dir: &Path, file_paths: &[PathBuf]) {
    for file_path in file_paths {
        let _ = ensure_thumbnail(thumbnails_dir, file_path);
    }
}

fn ensure_thumbnail(thumbnails_dir: &Path, file_path: &Path) -> Result<PathBuf, String> {
    let target = thumbnail_cache_path(thumbnails_dir, file_path);
    if target.is_file() {
        return Ok(target);
    }
    if generate_system_thumbnail(file_path, &target)? {
        return Ok(target);
    }

    let image = ImageReader::open(file_path)
        .map_err(|error| format!("Failed to read {}: {error}", file_path.display()))?
        .with_guessed_format()
        .map_err(|error| format!("Failed to read {}: {error}", file_path.display()))?
        .decode()
        .map_err(|error| format!("Failed to read {}: {error}", file_path.display()))?;
    let resized = image.thumbnail(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
    let mut temp_file = TempfileBuilder::new()
        .prefix("semanticgallery-thumb-")
        .suffix(".png")
        .tempfile_in(thumbnails_dir)
        .map_err(|error| format!("Failed to create a thumbnail cache file: {error}"))?;
    resized
        .write_to(temp_file.as_file_mut(), ImageFormat::Png)
        .map_err(|error| format!("Failed to write a thumbnail for {}: {error}", file_path.display()))?;
    temp_file
        .persist(&target)
        .map_err(|error| format!("Failed to save a thumbnail for {}: {error}", file_path.display()))?;
    Ok(target)
}

fn generate_system_thumbnail(file_path: &Path, target: &Path) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        let output_root = target.parent().ok_or_else(|| "The thumbnail cache path is invalid.".to_string())?;
        let temp_dir = tempfile::tempdir_in(output_root)
            .map_err(|error| format!("Failed to create a temporary thumbnail directory: {error}"))?;
        let status = Command::new("qlmanage")
            .args(["-t", "-s", &THUMBNAIL_SIZE.to_string(), "-o"])
            .arg(temp_dir.path())
            .arg(file_path)
            .status()
            .map_err(|error| format!("Failed to run the macOS thumbnail generator: {error}"))?;
        if !status.success() {
            return Ok(false);
        }
        let generated = fs::read_dir(temp_dir.path())
            .map_err(|error| format!("Failed to read the generated thumbnail directory: {error}"))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.is_file());
        let Some(generated_path) = generated else {
            return Ok(false);
        };
        fs::copy(&generated_path, target)
            .map_err(|error| format!("Failed to save the system thumbnail: {error}"))?;
        return Ok(true);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (file_path, target);
        Ok(false)
    }
}

fn delete_thumbnail(thumbnail_path: PathBuf) {
    let _ = fs::remove_file(thumbnail_path);
}

fn read_image_timestamp(file_path: &Path) -> Option<String> {
    let handle = File::open(file_path).ok()?;
    let mut reader = BufReader::new(handle);
    let exif = ExifReader::new().read_from_container(&mut reader).ok()?;
    for tag in [Tag::DateTimeOriginal, Tag::DateTimeDigitized, Tag::DateTime] {
        let field = exif.get_field(tag, In::PRIMARY)?;
        let raw = field.display_value().with_unit(&exif).to_string();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(parsed) = DateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S") {
            return Some(parsed.with_timezone(&Local).format("%Y-%m-%d %H:%M:%S").to_string());
        }
        return Some(trimmed.to_string());
    }
    None
}

fn format_local_time(time: SystemTime) -> String {
    let date_time: DateTime<Local> = DateTime::<Local>::from(time);
    date_time.format("%Y-%m-%d %H:%M:%S").to_string()
}

fn write_uploaded_image(bytes: &[u8], filename: Option<&str>) -> Result<NamedTempFile, String> {
    let stem = filename
        .and_then(|value| Path::new(value).file_stem())
        .and_then(|value| value.to_str())
        .unwrap_or("pasted-image")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    let safe_stem = stem.trim_matches(['-', '_']).to_string();
    let safe_stem = if safe_stem.is_empty() {
        "pasted-image".to_string()
    } else {
        safe_stem
    };

    let dynamic = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| "Unsupported image payload.".to_string())?
        .decode()
        .map_err(|_| "Unsupported image payload.".to_string())?;
    let mut temp_file = TempfileBuilder::new()
        .prefix(&format!("semanticgallery-{safe_stem}-"))
        .suffix(".jpg")
        .tempfile()
        .map_err(|error| format!("Failed to create a temporary image file: {error}"))?;
    let rgb = DynamicImage::ImageRgb8(dynamic.to_rgb8());
    let mut encoder = JpegEncoder::new_with_quality(temp_file.as_file_mut(), 92);
    encoder
        .encode_image(&rgb)
        .map_err(|_| "Unsupported image payload.".to_string())?;
    Ok(temp_file)
}

trait PathDisplayExt {
    fn as_posix(&self) -> String;
    fn expand_home(&self) -> Result<PathBuf, String>;
}

impl PathDisplayExt for Path {
    fn as_posix(&self) -> String {
        self.to_string_lossy().replace('\\', "/")
    }

    fn expand_home(&self) -> Result<PathBuf, String> {
        let text = self.as_posix();
        if text == "~" || text.starts_with("~/") {
            let home = std::env::var("HOME")
                .map(PathBuf::from)
                .map_err(|error| format!("Failed to resolve HOME: {error}"))?;
            if text == "~" {
                return Ok(home);
            }
            return Ok(home.join(text.trim_start_matches("~/")));
        }
        Ok(self.to_path_buf())
    }
}

impl PathDisplayExt for PathBuf {
    fn as_posix(&self) -> String {
        self.as_path().as_posix()
    }

    fn expand_home(&self) -> Result<PathBuf, String> {
        self.as_path().expand_home()
    }
}

async fn run_blocking<T, F>(desktop: State<'_, DesktopCoreState>, app: AppHandle, work: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce(&mut DesktopCore, &AppHandle) -> Result<T, String> + Send + 'static,
{
    let state = desktop.inner.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut core = state
            .lock()
            .map_err(|_| "The desktop workspace lock is unavailable.".to_string())?;
        work(&mut core, &app)
    })
    .await
    .map_err(|error| format!("The desktop workspace task did not finish: {error}"))?
}

#[tauri::command]
pub async fn desktop_runtime_status(app: AppHandle, desktop: State<'_, DesktopCoreState>) -> Result<RuntimeStatusPayload, String> {
    run_blocking(desktop, app, |core, app| core.runtime_status(app)).await
}

#[tauri::command]
pub async fn desktop_select_folder(
    app: AppHandle,
    desktop: State<'_, DesktopCoreState>,
    sidecar: State<'_, SidecarState>,
    args: SelectFolderArgs,
) -> Result<RefreshPayload, String> {
    let base_url = ensure_sidecar(&app, sidecar.inner())?;
    run_blocking(desktop, app, move |core, app| core.select_folder(app, &base_url, &args.folder_path)).await
}

#[tauri::command]
pub async fn desktop_refresh_folder(
    app: AppHandle,
    desktop: State<'_, DesktopCoreState>,
    sidecar: State<'_, SidecarState>,
    args: RefreshFolderArgs,
) -> Result<RefreshPayload, String> {
    let base_url = ensure_sidecar(&app, sidecar.inner())?;
    let lightweight = args.lightweight.unwrap_or(false);
    run_blocking(desktop, app, move |core, app| core.refresh_folder(app, &base_url, lightweight)).await
}

#[tauri::command]
pub async fn desktop_search_text(
    app: AppHandle,
    desktop: State<'_, DesktopCoreState>,
    sidecar: State<'_, SidecarState>,
    args: SearchTextArgs,
) -> Result<SearchPayload, String> {
    let base_url = ensure_sidecar(&app, sidecar.inner())?;
    run_blocking(desktop, app, move |core, app| core.search_text(app, &base_url, &args.query_text, args.limit)).await
}

#[tauri::command]
pub async fn desktop_search_uploaded_image(
    app: AppHandle,
    desktop: State<'_, DesktopCoreState>,
    sidecar: State<'_, SidecarState>,
    args: SearchUploadedImageArgs,
) -> Result<SearchPayload, String> {
    let base_url = ensure_sidecar(&app, sidecar.inner())?;
    run_blocking(desktop, app, move |core, app| {
        core.search_uploaded_image(
            app,
            &base_url,
            &args.image_base64,
            args.filename.as_deref(),
            args.limit,
        )
    })
    .await
}

#[tauri::command]
pub async fn desktop_search_similar(
    app: AppHandle,
    desktop: State<'_, DesktopCoreState>,
    sidecar: State<'_, SidecarState>,
    args: SearchSimilarArgs,
) -> Result<SearchPayload, String> {
    let base_url = ensure_sidecar(&app, sidecar.inner())?;
    run_blocking(desktop, app, move |core, app| core.search_similar(app, &base_url, &args.relative_path, args.limit))
        .await
}

#[tauri::command]
pub async fn desktop_metadata(
    app: AppHandle,
    desktop: State<'_, DesktopCoreState>,
    args: MetadataArgs,
) -> Result<MetadataPayload, String> {
    run_blocking(desktop, app, move |core, app| core.metadata(app, &args.relative_path)).await
}

#[tauri::command]
pub async fn desktop_delete_image(
    app: AppHandle,
    desktop: State<'_, DesktopCoreState>,
    args: DeleteImageArgs,
) -> Result<DeletePayload, String> {
    run_blocking(desktop, app, move |core, app| core.delete_image(app, &args.relative_path)).await
}

#[tauri::command]
pub async fn desktop_delete_images(
    app: AppHandle,
    desktop: State<'_, DesktopCoreState>,
    args: DeleteImagesArgs,
) -> Result<BatchDeletePayload, String> {
    run_blocking(desktop, app, move |core, app| core.delete_images(app, &args.paths)).await
}

#[tauri::command]
pub async fn desktop_run_stage2(
    app: AppHandle,
    desktop: State<'_, DesktopCoreState>,
    sidecar: State<'_, SidecarState>,
) -> Result<RuntimeStatusPayload, String> {
    let base_url = ensure_sidecar(&app, sidecar.inner())?;
    run_blocking(desktop, app, move |core, app| core.run_stage2(app, &base_url)).await
}

#[cfg(test)]
mod tests {
    use super::{build_scan_signature, normalize_vector, scan_progress_snapshot, ActiveSearchView};

    #[test]
    fn normalize_vector_rejects_zero_vector() {
        assert!(normalize_vector(&[0.0, 0.0]).is_none());
    }

    #[test]
    fn search_view_uses_cached_normalized_vectors() {
        let view = ActiveSearchView::from_rows(vec![
            ("/tmp/a.jpg".into(), vec![1.0, 0.0]),
            ("/tmp/b.jpg".into(), vec![0.0, 1.0]),
        ]);
        let results = view.search(&[1.0, 0.0], 1, None);
        assert_eq!(results, vec!["/tmp/a.jpg".to_string()]);
    }

    #[test]
    fn build_scan_signature_changes_when_metadata_changes() {
        let first = build_scan_signature(&[("cat.jpg".into(), 10, 11)]);
        let second = build_scan_signature(&[("cat.jpg".into(), 10, 12)]);
        assert_ne!(first, second);
    }

    #[test]
    fn scan_progress_snapshot_reports_checked_files_without_fake_totals() {
        let snapshot = scan_progress_snapshot(512, 381);
        assert_eq!(snapshot.status, "running");
        assert_eq!(snapshot.phase, "scan");
        assert_eq!(snapshot.current, 512);
        assert_eq!(snapshot.total, 0);
        assert_eq!(
            snapshot.message,
            "Scanning the selected folder for index updates. Checked 512 files and found 381 supported images."
        );
    }
}
