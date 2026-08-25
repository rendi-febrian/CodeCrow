package org.rostilos.codecrow.mcp.generic;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class LocalRepoClientTest {

    @TempDir
    Path repository;

    @Test
    void matchingTargetBranchAndRevisionReadFromLocalSnapshot() throws Exception {
        Files.createDirectories(repository.resolve("src"));
        Files.writeString(repository.resolve("src/App.java"), "class App {}\n");
        VcsMcpClient remote = mock(VcsMcpClient.class);
        LocalRepoClient client = new LocalRepoClient(
                remote,
                repository.toString(),
                "team",
                "repo",
                "main",
                "target-head-sha");

        assertThat(client.getBranchFileContent("team", "repo", "main", "src/App.java"))
                .isEqualTo("class App {}\n");
        assertThat(client.getBranchFileContent(
                "team", "repo", "target-head-sha", "src/App.java"))
                .isEqualTo("class App {}\n");
        verifyNoInteractions(remote);
    }

    @Test
    void nonTargetRevisionAndProviderOperationsRemainRemote() throws Exception {
        VcsMcpClient remote = mock(VcsMcpClient.class);
        when(remote.getBranchFileContent("team", "repo", "source-head", "src/App.java"))
                .thenReturn("remote source");
        when(remote.getPullRequestComments("team", "repo", "42"))
                .thenReturn(Map.of("comments", 1));
        LocalRepoClient client = new LocalRepoClient(
                remote,
                repository.toString(),
                "team",
                "repo",
                "main",
                "target-head-sha");

        assertThat(client.getBranchFileContent(
                "team", "repo", "source-head", "src/App.java"))
                .isEqualTo("remote source");
        assertThat(client.getPullRequestComments("team", "repo", "42"))
                .isEqualTo(Map.of("comments", 1));
        verify(remote).getBranchFileContent("team", "repo", "source-head", "src/App.java");
        verify(remote).getPullRequestComments("team", "repo", "42");
    }

    @Test
    void matchingRefForAnotherRepositoryNeverUsesThisJobsSnapshot() throws Exception {
        Files.createDirectories(repository.resolve("src"));
        Files.writeString(repository.resolve("src/App.java"), "tenant A source\n");
        VcsMcpClient remote = mock(VcsMcpClient.class);
        when(remote.getBranchFileContent(
                "other-team", "other-repo", "main", "src/App.java"))
                .thenReturn("tenant B source\n");
        LocalRepoClient client = new LocalRepoClient(
                remote,
                repository.toString(),
                "team",
                "repo",
                "main",
                "target-head-sha");

        assertThat(client.getBranchFileContent(
                "other-team", "other-repo", "main", "src/App.java"))
                .isEqualTo("tenant B source\n");
        verify(remote).getBranchFileContent(
                "other-team", "other-repo", "main", "src/App.java");
    }

    @Test
    void matchingTargetRefFallsBackToRemoteOnlyWhenPathIsAbsent() throws Exception {
        VcsMcpClient remote = mock(VcsMcpClient.class);
        when(remote.getBranchFileContent(
                "team", "repo", "target-head-sha", "generated/App.java"))
                .thenReturn("remote generated source");
        when(remote.getDirectoryByPath(
                "team", "repo", "target-head-sha", "generated"))
                .thenReturn("App.java");
        LocalRepoClient client = new LocalRepoClient(
                remote,
                repository.toString(),
                "team",
                "repo",
                "main",
                "target-head-sha");

        assertThat(client.getBranchFileContent(
                "team", "repo", "main", "generated/App.java"))
                .isEqualTo("remote generated source");
        assertThat(client.getDirectoryByPath("team", "repo", "main", "generated"))
                .isEqualTo("App.java");
        verify(remote).getBranchFileContent(
                "team", "repo", "target-head-sha", "generated/App.java");
        verify(remote).getDirectoryByPath(
                "team", "repo", "target-head-sha", "generated");
    }

    @Test
    void listsMatchingTargetDirectoriesLocallyInStableOrder() throws Exception {
        Files.createDirectories(repository.resolve("src/main"));
        Files.writeString(repository.resolve("README.md"), "read me");
        Files.writeString(repository.resolve("src/Z.java"), "class Z {}");
        Files.writeString(repository.resolve("src/A.java"), "class A {}");
        VcsMcpClient remote = mock(VcsMcpClient.class);
        LocalRepoClient client = new LocalRepoClient(
                remote,
                repository.toString(),
                "team",
                "repo",
                "main",
                "target-head-sha");

        assertThat(client.getRootDirectory("team", "repo", "main"))
                .isEqualTo("README.md\nsrc/");
        assertThat(client.getDirectoryByPath("team", "repo", "main", "src"))
                .isEqualTo("A.java\nZ.java\nmain/");
        verifyNoInteractions(remote);
    }

    @Test
    void rejectsPathsOutsideSnapshot() throws Exception {
        VcsMcpClient remote = mock(VcsMcpClient.class);
        LocalRepoClient client = new LocalRepoClient(
                remote,
                repository.toString(),
                "team",
                "repo",
                "main",
                "target-head-sha");

        assertThatThrownBy(() -> client.getBranchFileContent(
                "team", "repo", "main", "../secret.txt"))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("escapes the local snapshot");
        verifyNoInteractions(remote);
    }

    @Test
    void rejectsWrongLocalPathTypeWithoutRemoteFallback() throws Exception {
        Files.createDirectories(repository.resolve("src"));
        Files.writeString(repository.resolve("README.md"), "read me");
        VcsMcpClient remote = mock(VcsMcpClient.class);
        LocalRepoClient client = new LocalRepoClient(
                remote,
                repository.toString(),
                "team",
                "repo",
                "main",
                "target-head-sha");

        assertThatThrownBy(() -> client.getBranchFileContent(
                "team", "repo", "main", "src"))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("is not a file");
        assertThatThrownBy(() -> client.getDirectoryByPath(
                "team", "repo", "main", "README.md"))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("is not a directory");
        verifyNoInteractions(remote);
    }
}
