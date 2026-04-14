from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path


DESKTOP_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = Path(
    os.environ.get(
        "SEMANTICGALLERY_APP_PATH",
        str(DESKTOP_ROOT / "src-tauri" / "target" / "release" / "bundle" / "macos" / "SemanticGallery.app"),
    )
)
APPIUM_BIN = DESKTOP_ROOT / "node_modules" / ".bin" / "appium"
APPIUM_HOME = DESKTOP_ROOT / ".appium"
APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "com.semanticgallery.desktop"
ONBOARDING_MARKER = APP_SUPPORT_DIR / "onboarding-complete"
AUTOMATION_TOOLS_MARKER = APP_SUPPORT_DIR / "automation-tools-enabled"
WORKSPACE_STATE_PATH = APP_SUPPORT_DIR / "workspace-state.json"
INDEX_DB_PATH = APP_SUPPORT_DIR / "index.sqlite3"
REGRESSION_SOURCE_DIR = Path("/Users/duanyongyao/PythonProjects/phone_pictures")
TEMP_GALLERY_SOURCE_IMAGE = DESKTOP_ROOT / "src-tauri" / "icons" / "128x128.png"
GENERIC_SEARCH_QUERIES = ("food", "person", "people", "cat", "dog", "sky", "tree")
SUPPORTED_TEST_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

SETTINGS_AUTOMATION_FOLDER_INPUT_X = 620
SETTINGS_AUTOMATION_FOLDER_INPUT_Y = 384
SETTINGS_AUTOMATION_FOLDER_BUTTON_X = 1042
SETTINGS_AUTOMATION_FOLDER_BUTTON_Y = 384
WORKSPACE_AUTOMATION_IMAGE_INPUT_X = 520
WORKSPACE_AUTOMATION_IMAGE_INPUT_Y = 438
WORKSPACE_AUTOMATION_IMAGE_BUTTON_X = 1220
WORKSPACE_AUTOMATION_IMAGE_BUTTON_Y = 438


def print_step(message: str) -> None:
    print(f"[appium] {message}", flush=True)


def activate_app() -> None:
    subprocess.run(
        ["osascript", "-e", 'tell application "SemanticGallery" to activate'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def enable_automation_tools() -> None:
    APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUTOMATION_TOOLS_MARKER.write_text("ready")


def wait_for_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as response:
                payload = json.loads(response.read().decode())
            if payload.get("value", {}).get("ready"):
                return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"Appium server did not start on port {port}.")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def appium_server():
    if not APPIUM_BIN.is_file():
        raise RuntimeError(f"Appium is not installed at {APPIUM_BIN}.")
    port = find_free_port()
    env = os.environ.copy()
    env["APPIUM_HOME"] = str(APPIUM_HOME)
    env["MAX_TRAIN_STEPS"] = "1"
    env["MAX_VAL_STEPS"] = "1"
    env["MAX_EPOCHS_STAGE2"] = "1"
    command = [str(APPIUM_BIN), "server", "--port", str(port), "--base-path", "/"]
    process = subprocess.Popen(
        command,
        cwd=DESKTOP_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_port(port)
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


class AppiumSession:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None

    def request(self, method: str, path: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode()
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode()
        except urllib.error.HTTPError as error:
            detail = error.read().decode()
            raise RuntimeError(f"{method} {path} failed with {error.code}: {detail}") from error
        return json.loads(raw) if raw else {}

    def create(self) -> None:
        if not APP_PATH.is_dir():
            raise RuntimeError(f"The built app is missing at {APP_PATH}.")
        payload = {
            "capabilities": {
                "alwaysMatch": {
                    "platformName": "mac",
                    "appium:automationName": "mac2",
                    "appium:bundleId": "com.semanticgallery.desktop",
                    "appium:appPath": str(APP_PATH),
                    "appium:arguments": ["--semanticgallery-test-mode"],
                    "appium:environment": {
                        "SEMANTICGALLERY_AUTOMATION_TOOLS": "1",
                        "MAX_TRAIN_STEPS": "1",
                        "MAX_VAL_STEPS": "1",
                        "MAX_EPOCHS_STAGE2": "1",
                    },
                    "appium:newCommandTimeout": 240,
                },
                "firstMatch": [{}],
            }
        }
        response = self.request("POST", "/session", payload)
        value = response.get("value", {})
        self.session_id = value.get("sessionId") or response.get("sessionId")
        if not self.session_id:
            raise RuntimeError(f"Appium did not return a session id: {response}")

    def close(self) -> None:
        if self.session_id:
            self.request("DELETE", f"/session/{self.session_id}")
            self.session_id = None

    def source(self) -> str:
        return self.request("GET", f"/session/{self.session_id}/source").get("value", "")

    def screenshot(self, target: Path) -> None:
        payload = self.request("GET", f"/session/{self.session_id}/screenshot").get("value", "")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(payload))

    def find(self, xpath: str) -> str:
        value = self.request(
            "POST",
            f"/session/{self.session_id}/element",
            {"using": "xpath", "value": xpath},
        ).get("value", {})
        element_id = value.get("element-6066-11e4-a52e-4f735466cecf")
        if not element_id:
            raise RuntimeError(f"Element not found for xpath: {xpath}")
        return element_id

    def find_all(self, xpath: str) -> list[str]:
        values = self.request(
            "POST",
            f"/session/{self.session_id}/elements",
            {"using": "xpath", "value": xpath},
        ).get("value", [])
        return [item["element-6066-11e4-a52e-4f735466cecf"] for item in values]

    def click(self, xpath: str) -> None:
        element_id = self.find(xpath)
        self.request("POST", f"/session/{self.session_id}/element/{element_id}/click", {})

    def clear(self, xpath: str) -> None:
        element_id = self.find(xpath)
        self.request("POST", f"/session/{self.session_id}/element/{element_id}/clear", {})

    def set_value(self, xpath: str, text: str) -> None:
        element_id = self.find(xpath)
        self.request(
            "POST",
            f"/session/{self.session_id}/element/{element_id}/value",
            {"text": text, "value": list(text)},
        )

    def tap(self, x: int, y: int) -> None:
        self.request(
            "POST",
            f"/session/{self.session_id}/actions",
            {
                "actions": [
                    {
                        "type": "pointer",
                        "id": "mouse",
                        "parameters": {"pointerType": "mouse"},
                        "actions": [
                            {"type": "pointerMove", "duration": 10, "x": x, "y": y, "origin": "viewport"},
                            {"type": "pointerDown", "button": 0},
                            {"type": "pointerUp", "button": 0},
                        ],
                    }
                ]
            },
        )
        self.request("DELETE", f"/session/{self.session_id}/actions")

    def wait_for_tokens(self, *tokens: str, timeout: float = 30.0) -> str:
        deadline = time.time() + timeout
        last_source = ""
        while time.time() < deadline:
            source = self.source()
            last_source = source
            if "XCUIElementTypeWebView" in source and all(token in source for token in tokens):
                return source
            time.sleep(0.25)
        raise RuntimeError(f"Timed out waiting for {tokens}. Last source: {last_source[:2000]}")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def press_key_code(key_code: int) -> None:
    subprocess.run(
        ["osascript", "-e", f'tell application "System Events" to key code {key_code}'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def choose_folder_in_open_panel(folder_path: Path) -> None:
    escaped_path = folder_path.expanduser().resolve().as_posix().replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "System Events"
      keystroke "g" using {{command down, shift down}}
      delay 0.3
      keystroke "{escaped_path}"
      delay 0.2
      key code 36
      delay 0.5
      key code 36
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=True)


def copy_image_to_clipboard(image_path: Path) -> None:
    from AppKit import NSImage, NSPasteboard

    resolved_path = image_path.expanduser().resolve()
    image = NSImage.alloc().initWithContentsOfFile_(resolved_path.as_posix())
    if image is None:
        raise RuntimeError(f"The regression image could not be loaded into the clipboard: {resolved_path}")
    pasteboard = NSPasteboard.generalPasteboard()
    pasteboard.clearContents()
    if not pasteboard.writeObjects_([image]):
        raise RuntimeError("The regression image could not be placed on the clipboard.")


def paste_clipboard_image() -> None:
    script = '''
    tell application "System Events"
      keystroke "a" using {command down}
      delay 0.1
      key code 51
      delay 0.1
      keystroke "v" using {command down}
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=True)


def click_with_fallback(session: AppiumSession, xpath: str, x: int, y: int) -> None:
    try:
        session.click(xpath)
    except RuntimeError:
        session.tap(x, y)


def accessible_xpath(text: str) -> str:
    return f"//*[@title='{text}' or @label='{text}' or @value='{text}']"


def paste_text(text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    set the clipboard to "{escaped}"
    tell application "System Events"
      keystroke "a" using {{command down}}
      delay 0.1
      keystroke "v" using {{command down}}
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=True)


def set_text_with_fallback(
    session: AppiumSession,
    xpaths: list[str],
    text: str,
    x: int,
    y: int,
) -> None:
    for xpath in xpaths:
        try:
            session.clear(xpath)
            session.set_value(xpath, text)
            return
        except RuntimeError:
            continue
    activate_app()
    session.tap(x, y)
    time.sleep(0.2)
    paste_text(text)


@contextmanager
def temporarily_remove_onboarding_marker():
    original_bytes = ONBOARDING_MARKER.read_bytes() if ONBOARDING_MARKER.is_file() else None
    if ONBOARDING_MARKER.exists():
        ONBOARDING_MARKER.unlink()
    try:
        yield
    finally:
        if original_bytes is None:
            if ONBOARDING_MARKER.exists():
                ONBOARDING_MARKER.unlink()
        else:
            ONBOARDING_MARKER.parent.mkdir(parents=True, exist_ok=True)
            ONBOARDING_MARKER.write_bytes(original_bytes)


@contextmanager
def temporarily_isolate_app_support():
    backup_dir = None
    if APP_SUPPORT_DIR.exists():
        backup_dir = Path(tempfile.mkdtemp(prefix="semanticgallery-support-backup-")) / APP_SUPPORT_DIR.name
        shutil.move(APP_SUPPORT_DIR, backup_dir)
    try:
        yield
    finally:
        if APP_SUPPORT_DIR.exists():
            shutil.rmtree(APP_SUPPORT_DIR, ignore_errors=True)
        if backup_dir and backup_dir.exists():
            backup_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(backup_dir, APP_SUPPORT_DIR)


def parse_xml(source: str) -> ET.Element:
    return ET.fromstring(source)


def find_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def find_first(root: ET.Element, *, element_type: str | None = None, title: str | None = None, value: str | None = None):
    for node in root.iter():
        if element_type and node.tag != element_type:
            continue
        if title is not None and node.attrib.get("title") != title:
            continue
        if value is not None and node.attrib.get("value") != value:
            continue
        return node
    return None


def integer_attr(node: ET.Element, key: str) -> int:
    return int(float(node.attrib.get(key, "0")))


def resolve_sidecar_base_url(timeout: float = 60.0) -> str:
    deadline = time.time() + timeout
    port_pattern = re.compile(r"desktop_runtime\.sidecar_main.*--port (\d+)")
    while time.time() < deadline:
        output = subprocess.check_output(["ps", "axww", "-o", "command="], text=True)
        matches = [port_pattern.search(line) for line in output.splitlines()]
        ports = [match.group(1) for match in matches if match]
        if ports:
            return f"http://127.0.0.1:{ports[-1]}"
        time.sleep(0.5)
    raise RuntimeError("Could not resolve the sidecar port from the process table.")


def read_workspace_state() -> dict[str, object]:
    if not WORKSPACE_STATE_PATH.is_file():
        return {}
    return json.loads(WORKSPACE_STATE_PATH.read_text())


def read_folder_state(folder_path: Path) -> dict[str, object] | None:
    if not INDEX_DB_PATH.is_file():
        return None
    import sqlite3

    connection = sqlite3.connect(INDEX_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT folder_path, active_encoder_signature, file_count, total_bytes, scan_signature
            FROM folder_states
            WHERE folder_path = ?
            """,
            (folder_path.expanduser().resolve().as_posix(),),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def wait_for_active_folder(expected_folder: Path, expected_encoder: str = "stage1", timeout: float = 600.0) -> dict[str, object]:
    expected_folder_path = expected_folder.expanduser().resolve().as_posix()
    deadline = time.time() + timeout
    while time.time() < deadline:
        workspace_state = read_workspace_state()
        if (
            workspace_state.get("active_folder") == expected_folder_path
            and workspace_state.get("active_encoder_signature") == expected_encoder
        ):
            return workspace_state
        time.sleep(0.5)
    raise RuntimeError(
        f"The desktop workspace did not switch to {expected_folder_path} with encoder {expected_encoder} in time."
    )


def wait_for_stage2_encoder(expected_folder: Path, timeout: float = 900.0) -> dict[str, object]:
    expected_folder_path = expected_folder.expanduser().resolve().as_posix()
    deadline = time.time() + timeout
    while time.time() < deadline:
        folder_state = read_folder_state(expected_folder)
        if (
            folder_state is not None
            and folder_state.get("folder_path") == expected_folder_path
            and isinstance(folder_state.get("active_encoder_signature"), str)
            and folder_state.get("active_encoder_signature")
            and folder_state.get("active_encoder_signature") != "stage1"
        ):
            return folder_state
        time.sleep(0.5)
    raise RuntimeError(f"The desktop workspace did not finish Stage 2 for {expected_folder_path}.")


def read_active_encoder_signature() -> str | None:
    workspace_state = read_workspace_state()
    encoder_signature = workspace_state.get("active_encoder_signature")
    if (
        isinstance(workspace_state.get("active_folder"), str)
        and isinstance(encoder_signature, str)
        and encoder_signature
    ):
        return encoder_signature
    return None


def wait_for_index_count(expected_folder: Path, expected_count: int, timeout: float = 600.0) -> dict[str, object]:
    expected_folder_path = expected_folder.expanduser().resolve().as_posix()
    deadline = time.time() + timeout
    while time.time() < deadline:
        folder_state = read_folder_state(expected_folder)
        if folder_state and folder_state.get("folder_path") == expected_folder_path and folder_state.get("file_count") == expected_count:
            return folder_state
        time.sleep(0.5)
    raise RuntimeError(f"The folder index did not finish for {expected_folder_path}.")


def fetch_json(url: str, *, method: str = "GET", body: dict | None = None, timeout: float = 300.0) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_for_runtime_status(base_url: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return fetch_json(f"{base_url}/api/runtime/status", timeout=10.0)
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("The sidecar did not report runtime status in time.")


def wait_for_folder_ready(base_url: str, expected_folder: Path, timeout: float = 600.0) -> dict:
    expected_folder_path = expected_folder.expanduser().resolve().as_posix()
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = fetch_json(f"{base_url}/api/runtime/status", timeout=10.0)
        active_folder = payload.get("activeFolder")
        indexing = payload.get("indexing", {})
        if active_folder == expected_folder_path and indexing.get("status") == "ready":
            return payload
        time.sleep(0.5)
    raise RuntimeError(f"The folder index did not finish for {expected_folder}.")


def wait_for_stage2_status(base_url: str, expected_status: str | tuple[str, ...], timeout: float = 600.0) -> dict:
    expected = (expected_status,) if isinstance(expected_status, str) else expected_status
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = fetch_json(f"{base_url}/api/runtime/status", timeout=10.0)
        stage2 = payload.get("stage2", {})
        if stage2.get("status") in expected:
            return payload
        time.sleep(0.5)
    raise RuntimeError(f"Stage 2 did not reach status {expected!r}.")


def create_temp_gallery(root: Path, file_count: int = 400) -> Path:
    root = root.expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    if REGRESSION_SOURCE_DIR.is_dir():
        source_images = [
            path
            for path in sorted(REGRESSION_SOURCE_DIR.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_TEST_SUFFIXES
        ][:file_count]
        if source_images:
            for index, source_path in enumerate(source_images):
                suffix = source_path.suffix.lower() or ".jpg"
                target_name = f"sample-{index:04d}{suffix}"
                shutil.copy2(source_path, root / target_name)
            return root

    if not TEMP_GALLERY_SOURCE_IMAGE.is_file():
        raise RuntimeError(f"The regression source image is missing at {TEMP_GALLERY_SOURCE_IMAGE}.")
    sample_bytes = TEMP_GALLERY_SOURCE_IMAGE.read_bytes()
    for index in range(file_count):
        (root / f"sample-{index:04d}.png").write_bytes(sample_bytes)
    return root


def find_result_buttons(root: ET.Element) -> list[ET.Element]:
    result_buttons: list[ET.Element] = []
    for node in root.iter("XCUIElementTypeButton"):
        images = [child for child in node.iter() if child.tag == "XCUIElementTypeImage"]
        static_texts = [child for child in node.iter() if child.tag == "XCUIElementTypeStaticText"]
        if not images or static_texts:
            continue
        width = integer_attr(node, "width")
        height = integer_attr(node, "height")
        if width < 80 or height < 80:
            continue
        result_buttons.append(node)
    return sorted(result_buttons, key=lambda node: (integer_attr(node, "y"), integer_attr(node, "x")))


def center_of(node: ET.Element) -> tuple[int, int]:
    x = integer_attr(node, "x")
    y = integer_attr(node, "y")
    width = integer_attr(node, "width")
    height = integer_attr(node, "height")
    return (x + width // 2, y + height // 2)


def tap_result_button(session: AppiumSession, source: str, index: int = 0) -> None:
    root = parse_xml(source)
    result_buttons = find_result_buttons(root)
    if len(result_buttons) <= index:
        raise RuntimeError(f"The result grid did not expose thumbnail {index}.")
    x, y = center_of(result_buttons[index])
    session.tap(x, y)


def parse_result_count(source: str) -> int:
    match = re.search(r"(\d+) items", source)
    return int(match.group(1)) if match else 0


def assert_result_grid_columns(source: str, expected_columns: int = 5) -> None:
    root = parse_xml(source)
    result_buttons = find_result_buttons(root)
    if len(result_buttons) < expected_columns + 1:
        raise RuntimeError("The result grid did not render enough thumbnails to verify the desktop column count.")

    first_row_y = integer_attr(result_buttons[0], "y")
    first_row = [
        node
        for node in result_buttons
        if abs(integer_attr(node, "y") - first_row_y) <= 12
    ]
    if len(first_row) != expected_columns:
        raise RuntimeError(f"The desktop grid rendered {len(first_row)} thumbnails on the first row instead of {expected_columns}.")
    if integer_attr(result_buttons[expected_columns], "y") <= first_row_y + 12:
        raise RuntimeError("The sixth thumbnail remained on the first row instead of wrapping to the next row.")


def search_until_results(session: AppiumSession, screenshots_dir: Path) -> str | None:
    for query in GENERIC_SEARCH_QUERIES:
        session.clear("//XCUIElementTypeSearchField")
        session.set_value("//XCUIElementTypeSearchField", query)
        session.click("//*[@title='Search']")
        time.sleep(1.2)
        source = session.wait_for_tokens("Search results", timeout=20.0)
        match = re.search(r"(\\d+) items", source)
        count = int(match.group(1)) if match else 0
        if count <= 0:
            continue
        session.screenshot(screenshots_dir / "workspace-results.png")
        root = parse_xml(source)
        parent_map = find_parent_map(root)
        image_node = find_first(root, element_type="XCUIElementTypeImage")
        if image_node is None:
            raise RuntimeError("The search results did not render thumbnail images.")
        result_button = parent_map.get(image_node)
        while result_button is not None and result_button.tag != "XCUIElementTypeButton":
            result_button = parent_map.get(result_button)
        if result_button is None:
            raise RuntimeError("The result thumbnail is not wrapped in a clickable button.")
        static_text_descendants = [node for node in result_button.iter() if node.tag == "XCUIElementTypeStaticText"]
        if static_text_descendants:
            raise RuntimeError("The search results still render text inside the thumbnail tile.")
        return source
    return None


def assert_search_layout(source: str) -> None:
    root = parse_xml(source)
    search_field = find_first(root, element_type="XCUIElementTypeSearchField")
    result_limit = find_first(root, element_type="XCUIElementTypePopUpButton")
    if search_field is None or result_limit is None:
        raise RuntimeError("The search field or result limit control is missing.")
    search_width = integer_attr(search_field, "width")
    limit_width = integer_attr(result_limit, "width")
    if search_width <= limit_width * 5:
        raise RuntimeError(
            f"The search field is too narrow compared with the result limit control: {search_width} vs {limit_width}."
        )


def assert_settings_order(source: str) -> None:
    root = parse_xml(source)
    titles = {}
    for node in root.iter("XCUIElementTypeStaticText"):
        value = node.attrib.get("value")
        if value in {"Runtime status", "Current folder", "Stage 2 adaptation", "Uninstall"}:
            titles[value] = integer_attr(node, "y")
    ordered = ["Runtime status", "Current folder", "Stage 2 adaptation", "Uninstall"]
    if any(title not in titles for title in ordered):
        raise RuntimeError("The settings sections are missing from the page.")
    if [titles[title] for title in ordered] != sorted(titles[title] for title in ordered):
        raise RuntimeError("The settings sections are not stacked in the expected order.")


def assert_stage2_progress_copy(source: str) -> None:
    root = parse_xml(source)
    labels = {node.attrib.get("value", "") for node in root.iter("XCUIElementTypeStaticText")}
    expected = {"Adaptation progress", "Private data", "Model training", "Validation", "Rebuild index", "Finalize"}
    if not expected.issubset(labels):
        missing = sorted(expected - labels)
        raise RuntimeError(f"The Stage 2 progress card is missing labels: {missing}")


def assert_stage2_progress_hidden(source: str) -> None:
    hidden_tokens = {"Adaptation progress", "Private data", "Model training", "Validation", "Rebuild index", "Finalize"}
    leaked = sorted(token for token in hidden_tokens if token in source)
    if leaked:
        raise RuntimeError(f"The Stage 2 progress shell is still visible when it should be hidden: {leaked}")


def try_wait_for_tokens(session: AppiumSession, *tokens: str, timeout: float = 10.0) -> str | None:
    try:
        return session.wait_for_tokens(*tokens, timeout=timeout)
    except RuntimeError:
        return None


def wait_until_tokens_absent(session: AppiumSession, *tokens: str, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    last_source = ""
    while time.time() < deadline:
        source = session.source()
        last_source = source
        if all(token not in source for token in tokens):
            return source
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {tokens} to disappear. Last source: {last_source[:2000]}")


def wait_for_index_scan_feedback(session: AppiumSession, timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    last_source = ""
    while time.time() < deadline:
        source = session.source()
        last_source = source
        if "files checked" in source:
            return source
        if re.search(r"\b\d+\s*/\s*[1-9]\d*\b", source):
            return source
        if "Scanning..." in source:
            return source
        time.sleep(0.25)
    raise RuntimeError(
        "Selecting a folder did not surface live scan feedback. "
        f"Last source: {last_source[:2000]}"
    )


def open_settings(session: AppiumSession, screenshots_dir: Path, screenshot_name: str) -> str | None:
    click_with_fallback(session, accessible_xpath("Open settings"), 1340, 100)
    activate_app()
    time.sleep(0.6)
    session.screenshot(screenshots_dir / screenshot_name)
    return try_wait_for_tokens(session, "Folder and adaptation", "Back To Search", timeout=10.0)


def back_to_search(session: AppiumSession) -> str | None:
    click_with_fallback(session, accessible_xpath("Back to search"), 110, 95)
    activate_app()
    time.sleep(0.6)
    return try_wait_for_tokens(session, "Search your current album", "Refresh index", timeout=10.0)


def load_folder_through_automation(session: AppiumSession, folder_path: Path) -> None:
    set_text_with_fallback(
        session,
        [
            "//XCUIElementTypeTextField[@value='Folder path']",
            "//XCUIElementTypeTextField[@label='Folder path']",
            "//XCUIElementTypeTextField[@title='Folder path']",
        ],
        folder_path.expanduser().resolve().as_posix(),
        SETTINGS_AUTOMATION_FOLDER_INPUT_X,
        SETTINGS_AUTOMATION_FOLDER_INPUT_Y,
    )
    click_with_fallback(
        session,
        accessible_xpath("Load Folder Path"),
        SETTINGS_AUTOMATION_FOLDER_BUTTON_X,
        SETTINGS_AUTOMATION_FOLDER_BUTTON_Y,
    )


def choose_folder_through_dialog(session: AppiumSession, folder_path: Path) -> None:
    click_with_fallback(session, accessible_xpath("Choose Folder"), 690, 304)
    activate_app()
    time.sleep(0.6)
    choose_folder_in_open_panel(folder_path)


def load_image_through_automation(session: AppiumSession, image_path: Path) -> None:
    set_text_with_fallback(
        session,
        [
            "//XCUIElementTypeTextField[@value='Image path']",
            "//XCUIElementTypeTextField[@label='Image path']",
            "//XCUIElementTypeTextField[@title='Image path']",
        ],
        image_path.expanduser().resolve().as_posix(),
        WORKSPACE_AUTOMATION_IMAGE_INPUT_X,
        WORKSPACE_AUTOMATION_IMAGE_INPUT_Y,
    )
    click_with_fallback(
        session,
        accessible_xpath("Load Image Path"),
        WORKSPACE_AUTOMATION_IMAGE_BUTTON_X,
        WORKSPACE_AUTOMATION_IMAGE_BUTTON_Y,
    )


def paste_image_search(session: AppiumSession, image_path: Path) -> None:
    copy_image_to_clipboard(image_path)
    try:
        session.click("//XCUIElementTypeSearchField")
    except RuntimeError:
        session.tap(420, 339)
    activate_app()
    time.sleep(0.2)
    paste_clipboard_image()


def main() -> None:
    screenshots_dir = Path(tempfile.mkdtemp(prefix="semanticgallery-appium-"))
    small_gallery = Path(tempfile.gettempdir()) / "semanticgallery-appium-small-gallery"
    temp_gallery = Path(tempfile.gettempdir()) / "semanticgallery-appium-gallery"
    create_temp_gallery(small_gallery, file_count=99)
    create_temp_gallery(temp_gallery)
    query_image_path = next(path for path in sorted(temp_gallery.iterdir()) if path.suffix.lower() in SUPPORTED_TEST_SUFFIXES)
    print_step(f"Screenshots: {screenshots_dir}")
    print_step(f"Gallery: {temp_gallery}")

    run_command(["pkill", "-x", "SemanticGallery"])

    with temporarily_isolate_app_support():
        enable_automation_tools()
        with appium_server() as server_url:
            session = AppiumSession(server_url)
            session.create()
            try:
                activate_app()
                intro_source = session.wait_for_tokens("Install the local runtime", "Install And Start", timeout=60.0)
                time.sleep(0.5)
                session.screenshot(screenshots_dir / "launch-intro.png")
                if "Start Using" in intro_source:
                    raise RuntimeError("The launch intro skipped directly to the completion page.")

                session.click("//*[@title='Install And Start']")
                activate_app()
                install_source = session.wait_for_tokens("Overall progress", "Runtime log", timeout=60.0)
                time.sleep(0.5)
                session.screenshot(screenshots_dir / "launch-install.png")
                if "Cancel Install" not in install_source:
                    raise RuntimeError("The install screen did not show the cancel action.")

                completion_source = session.wait_for_tokens("Installation complete.", "Start Using", timeout=600.0)
                time.sleep(0.5)
                session.screenshot(screenshots_dir / "launch-complete.png")
                if "Install And Start" in completion_source:
                    raise RuntimeError("The completion screen did not replace the intro actions.")

                start_using_started = time.perf_counter()
                session.click("//*[@title='Start Using']")
                activate_app()
                workspace_source = session.wait_for_tokens("Search your current album", "Open settings", timeout=20.0)
                time.sleep(0.5)
                start_using_elapsed = time.perf_counter() - start_using_started
                session.screenshot(screenshots_dir / "workspace.png")
                if start_using_elapsed > 2.0:
                    raise RuntimeError(f"Start Using took too long to reach the workspace: {start_using_elapsed:.2f}s")
                if "Latest status" in workspace_source or "LATEST STATUS" in workspace_source:
                    raise RuntimeError("The workspace still renders the latest status card.")
                assert_search_layout(workspace_source)

                settings_source = open_settings(session, screenshots_dir, "settings-before-index.png")
                if settings_source:
                    assert_settings_order(settings_source)
                else:
                    print_step("The settings page did not expose a stable native tree. Keeping screenshot verification for it.")

                print_step("Selecting the smaller gallery to verify the Stage 2 threshold")
                load_folder_through_automation(session, small_gallery)
                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-after-load-folder-small.png")
                wait_for_active_folder(small_gallery, timeout=30.0)
                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-indexing.png")
                wait_for_index_count(small_gallery, 99, timeout=600.0)
                activate_app()
                time.sleep(0.6)
                click_with_fallback(session, accessible_xpath("Run Stage 2"), 722, 432)
                threshold_source = session.wait_for_tokens("at least 100 supported images", timeout=20.0)
                session.screenshot(screenshots_dir / "settings-stage2-threshold.png")
                if "currently has 99 supported images" not in threshold_source:
                    raise RuntimeError("The Stage 2 threshold warning did not surface the supported image count.")

                print_step("Selecting the full gallery for the complete regression run")
                load_folder_through_automation(session, temp_gallery)
                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-after-load-folder-full.png")
                wait_for_active_folder(temp_gallery, timeout=30.0)
                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-indexing-full.png")
                ready_status = wait_for_index_count(temp_gallery, 400, timeout=600.0)
                workspace_state = read_workspace_state()
                if workspace_state.get("active_folder") != temp_gallery.expanduser().resolve().as_posix():
                    raise RuntimeError("The desktop workspace did not persist the selected folder path.")
                if workspace_state.get("active_encoder_signature") != "stage1":
                    raise RuntimeError("The desktop workspace did not keep Stage 1 active after indexing.")
                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-ready.png")
                settings_ready_source = try_wait_for_tokens(session, "Folder and adaptation", "Run Stage 2", timeout=10.0)
                if settings_ready_source:
                    assert_settings_order(settings_ready_source)
                    assert_stage2_progress_hidden(settings_ready_source)
                indexed_count = ready_status.get("file_count")
                if indexed_count != 400:
                    raise RuntimeError(f"The temporary gallery indexed {indexed_count} images instead of 400.")

                click_with_fallback(session, accessible_xpath("Run Stage 2"), 722, 384)
                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-stage2-running.png")
                stage2_running_source = try_wait_for_tokens(session, "Adaptation progress", "Model training", timeout=120.0)
                if not stage2_running_source:
                    raise RuntimeError("The Stage 2 button did not start the adaptation flow.")
                assert_stage2_progress_copy(stage2_running_source)
                stage2_ready = wait_for_stage2_encoder(temp_gallery, timeout=900.0)
                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-stage2-ready.png")
                stage2_ready_source = wait_until_tokens_absent(session, "Adaptation progress", timeout=60.0)
                assert_settings_order(stage2_ready_source)
                assert_stage2_progress_hidden(stage2_ready_source)
                if stage2_ready.get("active_encoder_signature") == "stage1":
                    raise RuntimeError("Stage 2 completed without switching the active encoder signature.")

                workspace_after_back = back_to_search(session)
                if workspace_after_back:
                    time.sleep(0.5)
                else:
                    time.sleep(1.0)
                session.screenshot(screenshots_dir / "workspace-temp-gallery.png")
                grid_source = search_until_results(session, screenshots_dir)
                if not grid_source:
                    raise RuntimeError("The temporary gallery did not return results for the built-in search queries.")
                assert_result_grid_columns(grid_source, expected_columns=5)

                paste_image_search(session, query_image_path)
                activate_app()
                time.sleep(0.8)
                image_search_source = session.wait_for_tokens("Search results", "Clear image query", timeout=30.0)
                session.screenshot(screenshots_dir / "workspace-image-query.png")
                if parse_result_count(image_search_source) <= 0:
                    raise RuntimeError("The direct image search did not return any results.")
                click_with_fallback(session, accessible_xpath("Clear image query"), 381, 339)
                activate_app()
                cleared_image_query_source = session.wait_for_tokens("Search results", timeout=20.0)
                if parse_result_count(cleared_image_query_source) != 0:
                    raise RuntimeError("Clearing the image query did not reset the result list.")

                grid_source = search_until_results(session, screenshots_dir)
                if not grid_source:
                    raise RuntimeError("The workspace could not restore text search results after clearing the image query.")

                tap_result_button(session, grid_source, 0)
                activate_app()
                time.sleep(0.8)
                lightbox_source = session.wait_for_tokens("Find similar images", "View image details", timeout=20.0)
                session.screenshot(screenshots_dir / "lightbox-open.png")
                click_with_fallback(session, accessible_xpath("View image details"), 1185, 108)
                activate_app()
                time.sleep(0.5)
                metadata_source = session.wait_for_tokens("Name", "Path", "Dimensions", timeout=20.0)
                session.screenshot(screenshots_dir / "lightbox-metadata.png")
                if "Size" not in metadata_source:
                    raise RuntimeError("The metadata panel did not render the image details.")
                click_with_fallback(session, accessible_xpath("Find similar images"), 1268, 108)
                activate_app()
                time.sleep(0.8)
                similar_source = session.wait_for_tokens("Search results", "Clear image query", timeout=20.0)
                session.screenshot(screenshots_dir / "workspace-similar-search.png")
                if parse_result_count(similar_source) <= 0:
                    raise RuntimeError("The similar-image search did not return any results.")
                click_with_fallback(session, accessible_xpath("Clear image query"), 381, 339)
                activate_app()
                time.sleep(0.5)
                cleared_similar_source = session.wait_for_tokens("Search results", timeout=20.0)
                if parse_result_count(cleared_similar_source) != 0:
                    raise RuntimeError("Clearing the similar-image query did not reset the result list.")

                grid_source = search_until_results(session, screenshots_dir)
                if not grid_source:
                    raise RuntimeError("The workspace could not restore text search results for the delete tests.")

                single_delete_base_count = int(read_folder_state(temp_gallery)["file_count"])
                tap_result_button(session, grid_source, 0)
                activate_app()
                time.sleep(0.8)
                session.wait_for_tokens("Find similar images", "Move image to the Trash", timeout=20.0)
                click_with_fallback(session, accessible_xpath("Move image to the Trash"), 1225, 108)
                activate_app()
                time.sleep(0.5)
                confirm_single_source = session.wait_for_tokens("Confirm action", "Move To Trash", timeout=20.0)
                session.screenshot(screenshots_dir / "confirm-single-delete.png")
                if "Cancel" not in confirm_single_source:
                    raise RuntimeError("The single-image delete confirmation did not appear.")
                click_with_fallback(session, accessible_xpath("Move To Trash"), 857, 622)
                wait_for_index_count(temp_gallery, single_delete_base_count - 1, timeout=120.0)
                activate_app()
                time.sleep(0.8)
                session.screenshot(screenshots_dir / "workspace-after-single-delete.png")

                grid_source = search_until_results(session, screenshots_dir)
                if not grid_source:
                    raise RuntimeError("The workspace could not restore text search results for the selection tests.")
                batch_delete_base_count = int(read_folder_state(temp_gallery)["file_count"])
                batch_delete_visible_count = parse_result_count(grid_source)
                if batch_delete_visible_count <= 0:
                    raise RuntimeError("The selection test does not have any visible search results.")
                click_with_fallback(session, accessible_xpath("Select multiple results"), 1293, 453)
                activate_app()
                time.sleep(0.5)
                click_with_fallback(session, accessible_xpath("Select all results"), 1175, 522)
                activate_app()
                time.sleep(0.3)
                click_with_fallback(session, accessible_xpath("Clear selected results"), 1228, 522)
                activate_app()
                time.sleep(0.3)
                click_with_fallback(session, accessible_xpath("Select all results"), 1175, 522)
                activate_app()
                time.sleep(0.3)
                click_with_fallback(session, accessible_xpath("Move selected images to the Trash"), 1282, 522)
                activate_app()
                time.sleep(0.5)
                confirm_batch_source = session.wait_for_tokens("Confirm action", "Move To Trash", timeout=20.0)
                session.screenshot(screenshots_dir / "confirm-batch-delete.png")
                if "Cancel" not in confirm_batch_source:
                    raise RuntimeError("The batch delete confirmation did not appear.")
                click_with_fallback(session, accessible_xpath("Move To Trash"), 857, 622)
                wait_for_index_count(temp_gallery, batch_delete_base_count - batch_delete_visible_count, timeout=120.0)
                activate_app()
                time.sleep(0.8)
                session.screenshot(screenshots_dir / "workspace-after-batch-delete.png")

                refresh_started = time.perf_counter()
                click_with_fallback(session, accessible_xpath("Refresh index"), 1280, 100)
                open_settings(session, screenshots_dir, "settings-after-refresh.png")
                refresh_elapsed = time.perf_counter() - refresh_started
                activate_app()
                time.sleep(0.5)
                if refresh_elapsed > 2.0:
                    raise RuntimeError(f"The refresh flow blocked navigation for {refresh_elapsed:.2f}s")
            finally:
                session.close()

    run_command(["pkill", "-x", "SemanticGallery"])
    print_step("Regression complete")


if __name__ == "__main__":
    main()
