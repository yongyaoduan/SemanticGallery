import SwiftUI

enum WorkspaceMetrics {
    static let controlHeight: CGFloat = 44
    static let controlCornerRadius: CGFloat = 16
    static let buttonSize: CGFloat = 44
    static let buttonCornerRadius: CGFloat = 14
}

enum WorkspaceDeleteContext {
    case selection
    case preview(filename: String)
}
