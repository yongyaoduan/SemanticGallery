import CoreServices
import Foundation

public protocol FolderChangeMonitoring: AnyObject, Sendable {
    func startMonitoring(folderURL: URL, onChange: @escaping @Sendable (Int) -> Void) throws
    func stopMonitoring()
}

public final class FolderChangeMonitor: FolderChangeMonitoring, @unchecked Sendable {
    private var stream: FSEventStreamRef?
    private var onChange: (@Sendable (Int) -> Void)?

    public init() {}

    deinit {
        stopMonitoring()
    }

    public func startMonitoring(folderURL: URL, onChange: @escaping @Sendable (Int) -> Void) throws {
        stopMonitoring()
        self.onChange = onChange

        let paths = [folderURL.path(percentEncoded: false)] as CFArray
        var context = FSEventStreamContext(
            version: 0,
            info: UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()),
            retain: nil,
            release: nil,
            copyDescription: nil
        )

        let flags = FSEventStreamCreateFlags(
            kFSEventStreamCreateFlagUseCFTypes
                | kFSEventStreamCreateFlagFileEvents
                | kFSEventStreamCreateFlagNoDefer
                | kFSEventStreamCreateFlagWatchRoot
        )

        guard let stream = FSEventStreamCreate(
            kCFAllocatorDefault,
            folderChangeMonitorCallback,
            &context,
            paths,
            FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            0.5,
            flags
        ) else {
            throw NSError(
                domain: "SemanticGalleryPersistence",
                code: 1001,
                userInfo: [NSLocalizedDescriptionKey: "Folder monitoring could not start."]
            )
        }

        self.stream = stream
        FSEventStreamSetDispatchQueue(stream, .main)

        if FSEventStreamStart(stream) == false {
            stopMonitoring()
            throw NSError(
                domain: "SemanticGalleryPersistence",
                code: 1002,
                userInfo: [NSLocalizedDescriptionKey: "Folder monitoring could not start."]
            )
        }
    }

    public func stopMonitoring() {
        guard let stream else {
            onChange = nil
            return
        }

        FSEventStreamStop(stream)
        FSEventStreamInvalidate(stream)
        FSEventStreamRelease(stream)
        self.stream = nil
        self.onChange = nil
    }

    fileprivate func handleEventBatch(count: Int) {
        onChange?(max(count, 1))
    }
}

private func folderChangeMonitorCallback(
    streamRef: ConstFSEventStreamRef,
    clientCallBackInfo: UnsafeMutableRawPointer?,
    numEvents: Int,
    eventPaths: UnsafeMutableRawPointer,
    eventFlags: UnsafePointer<FSEventStreamEventFlags>,
    eventIds: UnsafePointer<FSEventStreamEventId>
) {
    guard let clientCallBackInfo else {
        return
    }

    let monitor = Unmanaged<FolderChangeMonitor>.fromOpaque(clientCallBackInfo).takeUnretainedValue()
    monitor.handleEventBatch(count: numEvents)
}
