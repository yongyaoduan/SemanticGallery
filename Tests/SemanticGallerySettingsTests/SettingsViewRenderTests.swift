import AppKit
import SwiftUI
import Testing
@testable import SemanticGalleryUI

@MainActor
@Test
func museumActionButtonsRemainVisibleWhenTheWindowIsInactive() throws {
    /// Formal specification
    /// Preconditions:
    ///   1. The settings window is visible but not key, so controls render in the inactive macOS control state.
    ///   2. The caller uses the standard SemanticGallery settings action styling.
    /// Postconditions:
    ///   1. The primary actions remain visually discoverable to the caller.
    ///   2. The rendered button preserves both a visible accent surface and a readable light label.

    let hostingView = NSHostingView(
        rootView: MuseumActionButton("Choose Folder", action: {})
            .frame(width: 220, height: 44)
            .environment(\.controlActiveState, .inactive)
    )
    hostingView.frame = NSRect(x: 0, y: 0, width: 220, height: 44)
    hostingView.layoutSubtreeIfNeeded()
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))

    let appearance = try analyzeAppearance(of: hostingView)
    #expect(appearance.warmPixelFraction > 0.08)
    #expect(appearance.brightPixelFraction > 0.005)
}

@MainActor
@Test
func museumActionButtonsUseGoldWhenEnabledAndPaleGoldWhenDisabled() throws {
    /// Formal specification
    /// Preconditions:
    ///   1. The caller renders the standard SemanticGallery settings action styling.
    ///   2. The same action can be rendered in enabled or disabled state.
    /// Postconditions:
    ///   1. The enabled state is visually stronger and more gold than the disabled state.
    ///   2. The disabled state remains visibly gold rather than collapsing into a neutral or brown fill.

    let enabledView = NSHostingView(
        rootView: MuseumActionButton("Choose Folder", action: {})
            .frame(width: 220, height: 44)
    )
    enabledView.frame = NSRect(x: 0, y: 0, width: 220, height: 44)
    enabledView.layoutSubtreeIfNeeded()

    let disabledView = NSHostingView(
        rootView: MuseumActionButton("Choose Folder", action: {})
            .frame(width: 220, height: 44)
            .disabled(true)
    )
    disabledView.frame = NSRect(x: 0, y: 0, width: 220, height: 44)
    disabledView.layoutSubtreeIfNeeded()

    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))

    let enabledAppearance = try analyzeAppearance(of: enabledView)
    let disabledAppearance = try analyzeAppearance(of: disabledView)

    #expect(enabledAppearance.warmPixelFraction > 0.08)
    #expect(enabledAppearance.accentSaturation > disabledAppearance.accentSaturation + 0.08)
    #expect(enabledAppearance.brightPixelFraction > 0.005)
    #expect(disabledAppearance.warmPixelFraction > 0.08)
}

private struct ButtonAppearance {
    let warmPixelFraction: Double
    let brightPixelFraction: Double
    let accentBrightness: Double
    let accentSaturation: Double
}

@MainActor
private func analyzeAppearance(of view: NSView) throws -> ButtonAppearance {
    view.layoutSubtreeIfNeeded()
    let bounds = view.bounds.integral
    let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: bounds))
    bitmap.size = bounds.size
    view.cacheDisplay(in: bounds, to: bitmap)

    let width = bitmap.pixelsWide
    let height = bitmap.pixelsHigh
    let bytesPerRow = bitmap.bytesPerRow
    let bytesPerPixel = bitmap.bitsPerPixel / 8
    let data = try #require(bitmap.bitmapData)

    var visiblePixels = 0
    var accentPixels = 0
    var warmPixels = 0
    var brightPixels = 0
    var accentBrightnessSum = 0.0
    var accentSaturationSum = 0.0

    for y in 0..<height {
        for x in 0..<width {
            let offset = y * bytesPerRow + x * bytesPerPixel
            let red = Double(data[offset]) / 255.0
            let green = Double(data[offset + 1]) / 255.0
            let blue = Double(data[offset + 2]) / 255.0
            let alpha = Double(data[offset + 3]) / 255.0

            guard alpha > 0.25 else {
                continue
            }

            visiblePixels += 1

            let maximum = max(red, green, blue)
            let minimum = min(red, green, blue)
            let saturation = maximum == 0 ? 0 : (maximum - minimum) / maximum
            let brightness = maximum

            if saturation > 0.05, brightness > 0.35, brightness < 0.99 {
                accentPixels += 1
                accentBrightnessSum += brightness
                accentSaturationSum += saturation
            }

            if brightness > 0.55, red > blue + 0.05, red >= green, green >= blue {
                warmPixels += 1
            }

            if brightness > 0.92, saturation < 0.18 {
                brightPixels += 1
            }
        }
    }

    let pixelCount = Double(max(visiblePixels, 1))
    let accentPixelCount = Double(max(accentPixels, 1))
    return ButtonAppearance(
        warmPixelFraction: Double(warmPixels) / pixelCount,
        brightPixelFraction: Double(brightPixels) / pixelCount,
        accentBrightness: accentBrightnessSum / accentPixelCount,
        accentSaturation: accentSaturationSum / accentPixelCount
    )
}
