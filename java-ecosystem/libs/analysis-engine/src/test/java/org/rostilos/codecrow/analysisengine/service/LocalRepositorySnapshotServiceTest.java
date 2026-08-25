package org.rostilos.codecrow.analysisengine.service;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.rostilos.codecrow.core.model.vcs.VcsConnection;
import org.rostilos.codecrow.vcsclient.VcsClient;
import org.rostilos.codecrow.vcsclient.VcsClientProvider;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class LocalRepositorySnapshotServiceTest {

    @TempDir
    Path temporaryRoot;

    @Test
    void usesSuppliedTargetHeadWithoutResolvingMovingBranch() throws Exception {
        BranchArchiveService archiveService = mock(BranchArchiveService.class);
        VcsClientProvider provider = mock(VcsClientProvider.class);
        VcsConnection connection = mock(VcsConnection.class);
        LocalRepositorySnapshotService service = new LocalRepositorySnapshotService(
                archiveService,
                provider,
                temporaryRoot);

        var prepared = service.prepare(
                connection,
                "team",
                "repo",
                "main",
                "pinned-target-head").orElseThrow();

        assertThat(prepared.revision()).isEqualTo("pinned-target-head");
        verifyNoInteractions(provider);
        verify(archiveService).downloadAndExtractSnapshotToDirectory(
                connection,
                "team",
                "repo",
                "pinned-target-head",
                null,
                prepared.path());
        prepared.close();
    }

    @Test
    void preparesPinnedTargetHeadAndDeletesWorkspaceOnClose() throws Exception {
        BranchArchiveService archiveService = mock(BranchArchiveService.class);
        VcsClientProvider provider = mock(VcsClientProvider.class);
        VcsClient client = mock(VcsClient.class);
        VcsConnection connection = mock(VcsConnection.class);
        when(provider.getClient(connection)).thenReturn(client);
        when(client.getLatestCommitHash("team", "repo", "main"))
                .thenReturn("target-head-sha");
        LocalRepositorySnapshotService service = new LocalRepositorySnapshotService(
                archiveService,
                provider,
                temporaryRoot);

        var prepared = service.prepare(connection, "team", "repo", "main").orElseThrow();

        assertThat(prepared.path()).exists().isDirectory();
        assertThat(prepared.targetBranch()).isEqualTo("main");
        assertThat(prepared.revision()).isEqualTo("target-head-sha");
        assertThat(prepared.transport().path()).isEqualTo(prepared.path().toString());
        verify(archiveService).downloadAndExtractSnapshotToDirectory(
                connection,
                "team",
                "repo",
                "target-head-sha",
                null,
                prepared.path());

        Path snapshotPath = prepared.path();
        prepared.close();
        assertThat(snapshotPath).doesNotExist();
    }

    @Test
    void archiveFailureCleansPartialWorkspaceAndFallsBack() throws Exception {
        BranchArchiveService archiveService = mock(BranchArchiveService.class);
        VcsClientProvider provider = mock(VcsClientProvider.class);
        VcsClient client = mock(VcsClient.class);
        VcsConnection connection = mock(VcsConnection.class);
        when(provider.getClient(connection)).thenReturn(client);
        when(client.getLatestCommitHash("team", "repo", "main"))
                .thenReturn("target-head-sha");
        doThrow(new IOException("archive unavailable"))
                .when(archiveService)
                .downloadAndExtractSnapshotToDirectory(
                        any(VcsConnection.class),
                        anyString(),
                        anyString(),
                        anyString(),
                        isNull(),
                        any(Path.class));
        LocalRepositorySnapshotService service = new LocalRepositorySnapshotService(
                archiveService,
                provider,
                temporaryRoot);

        assertThat(service.prepare(connection, "team", "repo", "main")).isEmpty();
        try (var children = Files.list(temporaryRoot)) {
            assertThat(children).isEmpty();
        }
    }

    @Test
    void missingTargetBranchSkipsProviderAndArchive() {
        BranchArchiveService archiveService = mock(BranchArchiveService.class);
        VcsClientProvider provider = mock(VcsClientProvider.class);
        LocalRepositorySnapshotService service = new LocalRepositorySnapshotService(
                archiveService,
                provider,
                temporaryRoot);

        assertThat(service.prepare(mock(VcsConnection.class), "team", "repo", " ")).isEmpty();
        verifyNoInteractions(provider, archiveService);
    }
}
