import Foundation
import SemanticGalleryPersistence

public struct SearchCoordinator {
    private let database: LibraryDatabase
    private let encoderVersion: String

    public init(database: LibraryDatabase, encoderVersion: String = "stage1") {
        self.database = database
        self.encoderVersion = encoderVersion
    }

    public func search(folderAbsolutePath: String, query: String, limit: Int) throws -> [SearchAsset] {
        let rows: [FileInstanceRecord]
        if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            rows = try database.visibleFileInstances(inFolderAbsolutePath: folderAbsolutePath, limit: limit)
        } else {
            rows = try database.searchFileInstances(inFolderAbsolutePath: folderAbsolutePath, query: query, limit: limit)
        }

        return rows.map { row in
            SearchAsset(
                fileInstanceID: row.fileInstanceID,
                assetID: row.assetID,
                absolutePath: row.absolutePath,
                relativePath: row.relativePath,
                thumbnailPath: nil
            )
        }
    }

    public func folderSearchIndex(folderAbsolutePath: String) throws -> FolderSearchIndex {
        let rows = try database.embeddedFiles(
            inFolderAbsolutePath: folderAbsolutePath,
            encoderVersion: encoderVersion
        )

        return FolderSearchIndex(
            items: rows.map { row in
                SearchAsset(
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
        searchIndex: FolderSearchIndex,
        queryVector: [Double],
        limit: Int
    ) -> [SearchAsset] {
        searchIndex.search(queryVector: queryVector, limit: limit)
    }

    public func searchSimilar(
        folderAbsolutePath: String,
        resultID: Int64,
        limit: Int
    ) throws -> [SearchAsset] {
        try folderSearchIndex(folderAbsolutePath: folderAbsolutePath)
            .searchSimilar(resultID: resultID, limit: limit)
    }
}
