package org.rostilos.codecrow.mcp;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.rostilos.codecrow.mcp.generic.VcsMcpClient;
import org.rostilos.codecrow.mcp.generic.VcsMcpClientFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class McpToolsLocalRepoTest {

    @TempDir
    Path repository;

    @AfterEach
    void clearLocalRepositoryProperties() {
        System.clearProperty(McpTools.LOCAL_REPO_PATH_PROPERTY);
        System.clearProperty(McpTools.LOCAL_REPO_TARGET_BRANCH_PROPERTY);
        System.clearProperty(McpTools.LOCAL_REPO_REVISION_PROPERTY);
        System.clearProperty(McpTools.WORKSPACE_PROPERTY);
        System.clearProperty(McpTools.REPO_SLUG_PROPERTY);
    }

    @Test
    void directoryDispatchUsesLocalSnapshotAndKeepsRemoteClientForProviderTools() throws Exception {
        Files.createDirectories(repository.resolve("src/main"));
        Files.writeString(repository.resolve("src/App.java"), "class App {}");
        System.setProperty(McpTools.LOCAL_REPO_PATH_PROPERTY, repository.toString());
        System.setProperty(McpTools.LOCAL_REPO_TARGET_BRANCH_PROPERTY, "main");
        System.setProperty(McpTools.LOCAL_REPO_REVISION_PROPERTY, "target-head-sha");
        System.setProperty(McpTools.WORKSPACE_PROPERTY, "team");
        System.setProperty(McpTools.REPO_SLUG_PROPERTY, "repo");

        VcsMcpClientFactory factory = mock(VcsMcpClientFactory.class);
        VcsMcpClient remote = mock(VcsMcpClient.class);
        when(factory.createClient()).thenReturn(remote);
        when(remote.getPullRequestComments("team", "repo", "42"))
                .thenReturn(Map.of("comments", 1));
        McpTools tools = new McpTools(factory);

        @SuppressWarnings("unchecked")
        Map<String, Object> directoryResult = (Map<String, Object>) tools.execute(
                "getDirectoryByPath",
                Map.of(
                        "workspace", "team",
                        "projectKey", "repo",
                        "branch", "main",
                        "dirPath", "src"));
        @SuppressWarnings("unchecked")
        Map<String, Object> commentsResult = (Map<String, Object>) tools.execute(
                "getPullRequestComments",
                Map.of(
                        "workspace", "team",
                        "repoSlug", "repo",
                        "pullRequestId", "42"));

        assertThat(directoryResult).containsEntry("directoryContent", "App.java\nmain/");
        assertThat(commentsResult).containsEntry("comments", Map.of("comments", 1));
        verify(remote, never()).getDirectoryByPath("team", "repo", "main", "src");
        verify(remote).getPullRequestComments("team", "repo", "42");
    }

    @Test
    void repositoryReadToolsRejectArgumentsOutsideTheRequestBinding() throws Exception {
        System.setProperty(McpTools.WORKSPACE_PROPERTY, "team");
        System.setProperty(McpTools.REPO_SLUG_PROPERTY, "repo");
        VcsMcpClientFactory factory = mock(VcsMcpClientFactory.class);
        VcsMcpClient remote = mock(VcsMcpClient.class);
        when(factory.createClient()).thenReturn(remote);
        McpTools tools = new McpTools(factory);

        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) tools.execute(
                "getBranchFileContent",
                Map.of(
                        "workspace", "other-team",
                        "repoSlug", "other-repo",
                        "branch", "main",
                        "filePath", "src/App.java"));

        assertThat(result.get("error").toString())
                .contains("request-bound repository");
        verify(remote, never()).getBranchFileContent(
                "other-team", "other-repo", "main", "src/App.java");
    }
}
