package org.rostilos.codecrow.analysisengine.service;

import org.rostilos.codecrow.analysisengine.dto.request.ai.LocalRepositorySnapshot;
import org.rostilos.codecrow.core.model.vcs.VcsConnection;
import org.rostilos.codecrow.vcsclient.VcsClient;
import org.rostilos.codecrow.vcsclient.VcsClientProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Optional;

/**
 * Prepares an exact target-branch snapshot for repository-aware MCP tools.
 *
 * <p>Snapshot preparation is optional enrichment. Provider or filesystem
 * failures leave the caller on its existing provider-backed MCP path.</p>
 */
@Service
public class LocalRepositorySnapshotService {
    private static final Logger log = LoggerFactory.getLogger(LocalRepositorySnapshotService.class);
    private static final String SNAPSHOT_PREFIX = "codecrow-pr-review-";

    private final BranchArchiveService archiveService;
    private final VcsClientProvider vcsClientProvider;
    private final Path temporaryRoot;

    @Autowired
    public LocalRepositorySnapshotService(
            BranchArchiveService archiveService,
            VcsClientProvider vcsClientProvider
    ) {
        this(archiveService, vcsClientProvider, Path.of(System.getProperty("java.io.tmpdir")));
    }

    LocalRepositorySnapshotService(
            BranchArchiveService archiveService,
            VcsClientProvider vcsClientProvider,
            Path temporaryRoot
    ) {
        this.archiveService = archiveService;
        this.vcsClientProvider = vcsClientProvider;
        this.temporaryRoot = temporaryRoot;
    }

    public Optional<PreparedSnapshot> prepare(
            VcsConnection connection,
            String workspace,
            String repoSlug,
            String targetBranch
    ) {
        return prepare(connection, workspace, repoSlug, targetBranch, null);
    }

    public Optional<PreparedSnapshot> prepare(
            VcsConnection connection,
            String workspace,
            String repoSlug,
            String targetBranch,
            String targetHeadRevision
    ) {
        if (targetBranch == null || targetBranch.isBlank()) {
            log.warn("Skipping local MCP repository snapshot because the PR target branch is unavailable");
            return Optional.empty();
        }

        Path snapshotDirectory = null;
        try {
            String targetRevision = targetHeadRevision;
            if (targetRevision == null || targetRevision.isBlank()) {
                VcsClient client = vcsClientProvider.getClient(connection);
                targetRevision = client.getLatestCommitHash(workspace, repoSlug, targetBranch);
            }
            if (targetRevision == null || targetRevision.isBlank()) {
                log.warn(
                        "Skipping local MCP repository snapshot because target head could not be resolved: {}/{} @ {}",
                        workspace, repoSlug, targetBranch);
                return Optional.empty();
            }

            snapshotDirectory = Files.createTempDirectory(temporaryRoot, SNAPSHOT_PREFIX);
            archiveService.downloadAndExtractSnapshotToDirectory(
                    connection,
                    workspace,
                    repoSlug,
                    targetRevision,
                    null,
                    snapshotDirectory);

            log.info(
                    "Prepared local MCP repository snapshot: {}/{} target={} revision={} path={}",
                    workspace,
                    repoSlug,
                    targetBranch,
                    shortRevision(targetRevision),
                    snapshotDirectory);
            return Optional.of(new PreparedSnapshot(
                    snapshotDirectory,
                    targetBranch,
                    targetRevision));
        } catch (Exception failure) {
            log.warn(
                    "Local MCP repository snapshot unavailable for {}/{} target={}; continuing with provider-backed tools: {}",
                    workspace,
                    repoSlug,
                    targetBranch,
                    failure.getMessage());
            deleteTree(snapshotDirectory);
            return Optional.empty();
        }
    }

    private static String shortRevision(String revision) {
        return revision.length() > 12 ? revision.substring(0, 12) + "…" : revision;
    }

    private static void deleteTree(Path root) {
        if (root == null) {
            return;
        }
        try {
            if (!Files.exists(root)) {
                return;
            }
            try (var paths = Files.walk(root)) {
                for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                    Files.deleteIfExists(path);
                }
            }
        } catch (IOException cleanupFailure) {
            log.warn("Failed to remove local MCP repository snapshot {}: {}",
                    root, cleanupFailure.getMessage());
        }
    }

    public static final class PreparedSnapshot implements AutoCloseable {
        private final Path path;
        private final String targetBranch;
        private final String revision;

        private PreparedSnapshot(Path path, String targetBranch, String revision) {
            this.path = path;
            this.targetBranch = targetBranch;
            this.revision = revision;
        }

        public Path path() {
            return path;
        }

        public String targetBranch() {
            return targetBranch;
        }

        public String revision() {
            return revision;
        }

        public LocalRepositorySnapshot transport() {
            return new LocalRepositorySnapshot(path.toString(), targetBranch, revision);
        }

        @Override
        public void close() {
            deleteTree(path);
        }
    }
}
