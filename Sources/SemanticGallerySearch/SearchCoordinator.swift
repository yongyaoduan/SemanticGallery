import Foundation
import SemanticGalleryPersistence

public struct SearchCoordinator {
    private let database: LibraryDatabase
    private let encoderVersion: String

    public init(database: LibraryDatabase, encoderVersion: String = "stage1") {
        self.database = database
        self.encoderVersion = encoderVersion
    }

    public func search(folderAbsolutePath: String, query: String, limit: Int) throws -> [SearchAssetRecord] {
        let rows: [FileInstanceRecord]
        if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            rows = try database.visibleFileInstances(inFolderAbsolutePath: folderAbsolutePath, limit: limit)
        } else {
            rows = try database.searchFileInstances(inFolderAbsolutePath: folderAbsolutePath, query: query, limit: limit)
        }

        return rows.map { row in
            SearchAssetRecord(
                fileInstanceID: row.fileInstanceID,
                assetID: row.assetID,
                absolutePath: row.absolutePath,
                relativePath: row.relativePath,
                thumbnailPath: nil
            )
        }
    }

    public func activeSearchView(folderAbsolutePath: String) throws -> ActiveSearchView {
        let rows = try database.embeddedFiles(
            inFolderAbsolutePath: folderAbsolutePath,
            encoderVersion: encoderVersion
        )

        return ActiveSearchView(
            items: rows.map { row in
                SearchAssetRecord(
                    fileInstanceID: row.fileInstanceID,
                    assetID: row.assetID,
                    absolutePath: row.absolutePath,
                    relativePath: row.relativePath,
                    thumbnailPath: nil
                )
            },
            matrix: rows.map(\.vector)
        )
    }

    public func vectorSearch(
        activeView: ActiveSearchView,
        queryVector: [Double],
        limit: Int
    ) -> [SearchAssetRecord] {
        activeView.search(queryVector: queryVector, limit: limit)
    }

    public func searchSimilar(
        folderAbsolutePath: String,
        recordID: Int64,
        limit: Int
    ) throws -> [SearchAssetRecord] {
        try activeSearchView(folderAbsolutePath: folderAbsolutePath)
            .searchSimilar(recordID: recordID, limit: limit)
    }
}
