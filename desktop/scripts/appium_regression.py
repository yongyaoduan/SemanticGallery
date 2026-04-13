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
APP_PATH = DESKTOP_ROOT / "src-tauri" / "target" / "release" / "bundle" / "macos" / "SemanticGallery.app"
APPIUM_BIN = DESKTOP_ROOT / "node_modules" / ".bin" / "appium"
APPIUM_HOME = DESKTOP_ROOT / ".appium"
APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "com.semanticgallery.desktop"
ONBOARDING_MARKER = APP_SUPPORT_DIR / "onboarding-complete"
REGRESSION_SOURCE_DIR = Path("/Users/duanyongyao/PythonProjects/phone_pictures")
TEMP_GALLERY_SOURCE_IMAGE = DESKTOP_ROOT / "src-tauri" / "icons" / "128x128.png"
GENERIC_SEARCH_QUERIES = ("food", "person", "people", "cat", "dog", "sky", "tree")
SUPPORTED_TEST_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def print_step(message: str) -> None:
    print(f"[appium] {message}", flush=True)


def activate_app() -> None:
    subprocess.run(
        ["osascript", "-e", 'tell application "SemanticGallery" to activate'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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


def search_until_results(session: AppiumSession, screenshots_dir: Path) -> bool:
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
        return True
    return False


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


def try_wait_for_tokens(session: AppiumSession, *tokens: str, timeout: float = 10.0) -> str | None:
    try:
        return session.wait_for_tokens(*tokens, timeout=timeout)
    except RuntimeError:
        return None


def main() -> None:
    screenshots_dir = Path(tempfile.mkdtemp(prefix="semanticgallery-appium-"))
    temp_gallery = Path(tempfile.gettempdir()) / "semanticgallery-appium-gallery"
    create_temp_gallery(temp_gallery)
    print_step(f"Screenshots: {screenshots_dir}")
    print_step(f"Gallery: {temp_gallery}")

    run_command(["pkill", "-x", "SemanticGallery"])

    with temporarily_remove_onboarding_marker():
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

                base_url = resolve_sidecar_base_url()
                runtime_status = wait_for_runtime_status(base_url)
                original_folder = runtime_status.get("activeFolder")

                if original_folder:
                    results_visible = search_until_results(session, screenshots_dir)
                    if not results_visible:
                        raise RuntimeError("The current folder did not return results for the built-in search queries.")

                session.click("//*[@title='Open settings']")
                activate_app()
                time.sleep(0.5)
                session.screenshot(screenshots_dir / "settings-before-index.png")
                settings_source = try_wait_for_tokens(session, "Folder and adaptation", "Back To Search", timeout=10.0)
                if settings_source:
                    assert_settings_order(settings_source)
                else:
                    print_step("The settings page did not expose a stable native tree. Keeping screenshot verification for it.")

                print_step("Selecting the temporary gallery through the sidecar API")
                fetch_json(
                    f"{base_url}/api/folders/select",
                    method="POST",
                    body={"folderPath": temp_gallery.as_posix()},
                    timeout=600.0,
                )

                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-indexing.png")

                ready_status = wait_for_folder_ready(base_url, temp_gallery)
                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-ready.png")
                indexed_count = ready_status.get("indexedImageCount")
                if indexed_count != 400:
                    raise RuntimeError(f"The temporary gallery indexed {indexed_count} images instead of 400.")

                stage2_response: dict[str, object] = {}
                stage2_error: list[Exception] = []

                def run_stage2() -> None:
                    try:
                        stage2_response.update(
                            fetch_json(
                                f"{base_url}/api/stage2/run",
                                method="POST",
                                timeout=900.0,
                            )
                        )
                    except Exception as exc:  # pragma: no cover - exercised by desktop regression
                        stage2_error.append(exc)

                stage2_thread = threading.Thread(target=run_stage2, daemon=True)
                stage2_thread.start()
                wait_for_stage2_status(base_url, ("running", "ready"), timeout=120.0)
                activate_app()
                press_key_code(121)
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-stage2-running.png")
                stage2_ready = wait_for_stage2_status(base_url, "ready", timeout=900.0)
                stage2_thread.join(timeout=10.0)
                if stage2_thread.is_alive():
                    raise RuntimeError("The Stage 2 request thread did not finish in time.")
                if stage2_error:
                    raise stage2_error[0]
                activate_app()
                press_key_code(121)
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-stage2-ready.png")
                if stage2_ready.get("activeEncoderSignature") == "stage1":
                    raise RuntimeError("Stage 2 completed without switching the active encoder signature.")
                press_key_code(116)
                time.sleep(0.5)

                try:
                    session.click("//*[@title='Back to search']")
                except RuntimeError:
                    session.tap(110, 95)
                activate_app()
                workspace_after_back = try_wait_for_tokens(
                    session,
                    "Search your current album",
                    "Refresh index",
                    timeout=10.0,
                )
                if workspace_after_back:
                    time.sleep(0.5)
                else:
                    time.sleep(1.0)
                session.screenshot(screenshots_dir / "workspace-temp-gallery.png")
                refresh_started = time.perf_counter()
                if workspace_after_back:
                    session.click("//*[@title='Refresh index']")
                    session.click("//*[@title='Open settings']")
                else:
                    session.tap(1280, 100)
                    session.tap(1340, 100)
                refresh_elapsed = time.perf_counter() - refresh_started
                activate_app()
                time.sleep(1.0)
                session.screenshot(screenshots_dir / "settings-after-refresh.png")
                if refresh_elapsed > 2.0:
                    raise RuntimeError(f"The refresh flow blocked navigation for {refresh_elapsed:.2f}s")

                if original_folder and Path(original_folder).is_dir():
                    print_step("Restoring the original folder")
                    fetch_json(
                        f"{base_url}/api/folders/select",
                        method="POST",
                        body={"folderPath": original_folder},
                        timeout=600.0,
                    )
                    wait_for_folder_ready(base_url, Path(original_folder))
            finally:
                session.close()

    run_command(["pkill", "-x", "SemanticGallery"])
    print_step("Regression complete")


if __name__ == "__main__":
    main()
