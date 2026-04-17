# SemanticGallery Usage Workspace Redesign

## Goal

Redesign the usage workspace so it feels like a refined desktop photo tool rather than a settings-heavy control panel. The workspace should emphasize the image grid, keep text to a minimum, and separate browsing from batch selection in a way that feels immediate and natural.

## Scope

This redesign covers:

- the empty usage state when no folder has been chosen yet
- the main workspace header and search controls
- the browse state and select state split
- the result grid behavior
- the large-image preview overlay
- the delete confirmation experience
- the tests required to verify the new interaction model

This redesign does not change installation, folder preparation logic, model loading, search ranking, or settings architecture.

## Implementation Contract

The redesign should stay inside these files unless a small support type is required:

- `Sources/SemanticGallerySearch/UsageView.swift`
- `Sources/SemanticGallerySearch/WorkspaceStateStore.swift`
- `Sources/SemanticGalleryAppShell/SemanticGalleryController.swift`
- `Sources/SemanticGallerySearch/ThumbnailStore.swift` when preview image reuse needs a shared path
- `App/SemanticGalleryUITests/PhaseAFlowTests.swift`
- `Tests/SemanticGalleryAppShellTests/SemanticGalleryControllerTests.swift`

The workspace state needs explicit fields for the new interaction model:

- `isSelectionModeEnabled: Bool`
- `previewAssetID: Int64?`
- `isPreviewMetadataVisible: Bool`

The redesigned usage page should expose concrete accessibility identifiers so the UI tests can stop guessing:

- `workspace-empty-open-settings`
- `workspace-path-bar`
- `workspace-result-limit-picker`
- `workspace-search-button`
- `workspace-selection-mode-button`
- `workspace-selection-toggle-button`
- `workspace-delete-button`
- `workspace-preview-overlay`
- `workspace-preview-close-button`
- `workspace-preview-info-button`
- `workspace-preview-metadata`
- `workspace-preview-next-button`
- `workspace-preview-previous-button`
- `workspace-delete-confirmation`

## Product Direction

The page should remain inside the existing museum-paper visual system, but the workspace itself should become quieter and more image-led. The visual tone should feel archival, elegant, and calm without describing itself that way in copy. Buttons should rely on icons instead of visible text labels wherever the interaction is already familiar.

## Interaction Model

The workspace should have three user-facing states.

### Empty State

When no library folder is selected:

- show one short line that implies the workspace is waiting for a library
- show a single `Open Settings` entry point
- do not expose folder choice directly inside the usage page
- do not render `folder-preparation-progress-group`, status banners, or technical explanations here

### Browse State

Browse is the default state after a folder has been prepared.

- clicking a thumbnail opens the large-image preview
- the page header shows only the current path, the search field, the result count control, the search button, and the entry button for selection mode
- no extra title block, no folder/location chips, and no status copy appear in the workspace
- pasted-image search remains available through the main search field
- the preview should open from the same `SearchAssetRecord` array already used by `workspace-results-grid`

### Select State

Selection mode begins only when the user explicitly enters it.

- clicking a thumbnail toggles selection instead of opening preview
- the top action area switches into a compact icon-only selection toolbar
- one toggle button handles both `select all visible` and `clear all`
- delete stays disabled until at least one item is selected
- leaving selection mode clears the current selection so the workspace returns to a clean browse state
- entering selection mode closes any open preview and resets `isPreviewMetadataVisible` to `false`

## Layout

### Top Bar

The top bar should be compact and horizontal.

- the path appears as a single Finder-like path line with middle truncation when needed
- the path line should come from `selectedFolder.path(percentEncoded: false)` and keep the final folder name visible
- the search field takes the visual center and stays wide
- the result count uses a compact menu control
- the search action uses a distinct, polished icon button
- the selection entry button uses an icon, not a text label
- the visible controls should fit in one row at a window width of `1200` points

### Search Field

The search field should look like a refined paper strip rather than a generic form control.

- support text input
- support pasted image input
- show a small image token when an image query is present
- use a small icon to remove the pasted image
- avoid explanatory helper paragraphs inside the workspace
- keep the existing paste entry path in `PasteAwareSearchField`, but move all visible helper copy out of the main control

### Result Grid

- keep the grid at five columns with narrow spacing
- keep the existing `LazyVGrid` column count at `5`
- do not wrap thumbnails in extra card containers
- show selection state through a restrained highlight and corner mark only when selection mode is active
- keep loading states quiet and unobtrusive
- clicking a cell in browse mode should call preview open logic, not `toggleSelection`

## Preview Overlay

The large-image view should follow a Quick Look-like model.

- the image opens in a darkened full-window overlay
- the overlay should sit above the grid inside `UsageView` rather than open a separate window
- the image remains the primary focus
- left and right navigation stay minimal and low-noise
- `Escape` closes the overlay
- left and right arrow keys move between images
- clicking the dimmed background closes the overlay
- if the preview is showing the first or last visible item, the previous or next arrow should disable cleanly

### Preview Metadata

Metadata should be hidden by default.

- place an `info` icon button in the preview chrome
- tapping the `info` button reveals a light metadata sheet inside the overlay
- tapping it again hides the sheet
- the metadata sheet should open without moving the image itself

The metadata sheet should show:

- file name
- full path
- capture time when available from image metadata
- fallback time from file metadata when capture time is missing
- file size
- image dimensions

The metadata lookup order should be:

1. `kCGImagePropertyExifDateTimeOriginal`
2. `kCGImagePropertyTIFFDateTime`
3. file creation date
4. file content modification date

The metadata panel should never show internal implementation details such as ids, encoder versions, or index data.

## Delete Confirmation

The delete confirmation should feel deliberate and polished rather than generic.

- title the action as moving images to Trash
- show the selected file name for a single item
- show the first file name plus quantity for multiple items
- keep the destructive action explicit
- keep the cancel action easy to find
- match the museum-paper surface language instead of default system dialog styling
- the destructive button text should stay `Move to Trash`

After confirmation:

- files must be moved to Trash
- removed images disappear from the grid immediately
- the index must drop those records
- the in-memory result set must update in the same operation

## Accessibility And Keyboard

- every icon button still needs a clear accessibility label
- preview open, preview close, selection mode toggle, select-all toggle, and delete need keyboard coverage in tests
- preview navigation should respond to arrow keys
- preview close should respond to `Escape`
- the `info` icon needs an accessibility label that reads as image information, not punctuation

## Tests

The redesign must add or update real tests.

### Unit Tests

- selection mode state transitions
- leaving selection mode clears selection
- select-all toggle behavior
- preview item navigation logic
- metadata fallback behavior when capture time is absent
- preview closes when selection mode is entered
- preview metadata resets when the preview closes

### XCUITest

- empty state shows the new minimal entry point
- browse mode opens preview from a thumbnail click
- preview closes with `Escape`
- preview metadata opens from the info icon
- selection mode changes click behavior from preview to selection
- select-all toggle switches between full selection and cleared selection
- delete confirmation appears with the correct file context
- delete removes files, updates the grid, and leaves the index clean
- the installed-app run from `/Applications/SemanticGallery.app` covers preview, selection mode, and deletion

### Visual Validation

Capture fresh evidence for:

- empty usage state
- browse state
- selection state
- preview overlay
- preview metadata sheet
- delete confirmation

The verification commands for this redesign are:

- `cd /Users/duanyongyao/PythonProjects/SemanticGallery/repo && swift test --no-parallel`
- `cd /Users/duanyongyao/PythonProjects/SemanticGallery/repo && ./scripts/run-ui-tests.sh`
- targeted installed-app `xcodebuild test-without-building` runs for the preview, selection, and delete tests

## Success Criteria

The redesign is complete when:

- the usage page is visibly simpler and more image-forward
- browse mode and select mode are clearly distinct
- large-image preview works without entering selection mode
- metadata is available on demand through the preview info button
- batch selection and deletion remain fast and correct
- updated unit tests and XCUITests pass against the real app flow
