public struct RepairDetector {
    public let installStateStore: InstallStateStore

    public init(installStateStore: InstallStateStore) {
        self.installStateStore = installStateStore
    }

    public func requiresRepair() throws -> Bool {
        try installStateStore.isInstallationComplete() == false
    }
}
