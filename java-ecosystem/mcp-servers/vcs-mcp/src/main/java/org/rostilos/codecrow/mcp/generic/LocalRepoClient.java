package org.rostilos.codecrow.mcp.generic;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Provider-neutral VCS client adapter that serves one pinned repository tree
 * from disk while delegating provider operations to the normal remote client.
 */
public final class LocalRepoClient implements VcsMcpClient {
    private final VcsMcpClient remoteClient;
    private final Path repoRoot;
    private final Path realRepoRoot;
    private final String expectedWorkspace;
    private final String expectedRepoSlug;
    private final String targetBranch;
    private final String targetRevision;

    public LocalRepoClient(
            VcsMcpClient remoteClient,
            String repoRootPath,
            String expectedWorkspace,
            String expectedRepoSlug,
            String targetBranch,
            String targetRevision
    ) throws IOException {
        this.remoteClient = Objects.requireNonNull(remoteClient, "remoteClient");
        this.repoRoot = Path.of(repoRootPath).toAbsolutePath().normalize();
        if (!Files.isDirectory(repoRoot)) {
            throw new IOException("Local repository path is not a directory: " + repoRoot);
        }
        this.realRepoRoot = repoRoot.toRealPath();
        this.expectedWorkspace = Objects.requireNonNull(
                expectedWorkspace, "expectedWorkspace");
        this.expectedRepoSlug = Objects.requireNonNull(
                expectedRepoSlug, "expectedRepoSlug");
        this.targetBranch = targetBranch;
        this.targetRevision = targetRevision;
    }

    private boolean isLocalRef(
            String workspace,
            String repoSlug,
            String branchOrRevision
    ) {
        if (branchOrRevision == null || branchOrRevision.isBlank()) {
            return false;
        }
        return expectedWorkspace.equals(workspace)
                && expectedRepoSlug.equals(repoSlug)
                && (branchOrRevision.equals(targetBranch)
                || branchOrRevision.equals(targetRevision));
    }

    private Path resolveExistingPath(String relativePath) throws IOException {
        String path = relativePath == null ? "" : relativePath;
        Path candidate = repoRoot.resolve(path).normalize();
        if (!candidate.startsWith(repoRoot)) {
            throw new IOException("Repository path escapes the local snapshot: " + path);
        }
        if (!Files.exists(candidate, LinkOption.NOFOLLOW_LINKS)) {
            throw new NoSuchFileException(path);
        }
        Path realCandidate = candidate.toRealPath();
        if (!realCandidate.startsWith(realRepoRoot)) {
            throw new IOException("Repository path escapes the local snapshot: " + path);
        }
        return realCandidate;
    }

    private String readDirectory(String dirPath) throws IOException {
        Path directory = resolveExistingPath(dirPath);
        if (!Files.isDirectory(directory)) {
            throw new IOException("Repository path is not a directory: " + dirPath);
        }
        try (var entries = Files.list(directory)) {
            return String.join("\n", entries
                    .sorted((left, right) -> left.getFileName().toString()
                            .compareTo(right.getFileName().toString()))
                    .map(entry -> entry.getFileName().toString()
                            + (Files.isDirectory(entry, LinkOption.NOFOLLOW_LINKS) ? "/" : ""))
                    .toList());
        }
    }

    @Override
    public String getProviderType() {
        return remoteClient.getProviderType();
    }

    @Override
    public String getPrNumber() throws IOException {
        return remoteClient.getPrNumber();
    }

    @Override
    public String getPullRequestTitle() throws IOException {
        return remoteClient.getPullRequestTitle();
    }

    @Override
    public String getPullRequestDescription() throws IOException {
        return remoteClient.getPullRequestDescription();
    }

    @Override
    public List<FileDiffInfo> getPullRequestChanges() throws IOException {
        return remoteClient.getPullRequestChanges();
    }

    @Override
    public List<Map<String, Object>> listRepositories(String workspace, Integer limit) throws IOException {
        return remoteClient.listRepositories(workspace, limit);
    }

    @Override
    public Map<String, Object> getRepository(String workspace, String repoSlug) throws IOException {
        return remoteClient.getRepository(workspace, repoSlug);
    }

    @Override
    public List<Map<String, Object>> getPullRequests(
            String workspace,
            String repoSlug,
            String state,
            Integer limit
    ) throws IOException {
        return remoteClient.getPullRequests(workspace, repoSlug, state, limit);
    }

    @Override
    public Map<String, Object> createPullRequest(
            String workspace,
            String repoSlug,
            String title,
            String description,
            String sourceBranch,
            String targetBranch,
            List<String> reviewers
    ) throws IOException {
        return remoteClient.createPullRequest(
                workspace, repoSlug, title, description, sourceBranch, targetBranch, reviewers);
    }

    @Override
    public Map<String, Object> getPullRequest(
            String workspace,
            String repoSlug,
            String pullRequestId
    ) throws IOException {
        return remoteClient.getPullRequest(workspace, repoSlug, pullRequestId);
    }

    @Override
    public Map<String, Object> updatePullRequest(
            String workspace,
            String repoSlug,
            String pullRequestId,
            String title,
            String description
    ) throws IOException {
        return remoteClient.updatePullRequest(workspace, repoSlug, pullRequestId, title, description);
    }

    @Override
    public Object getPullRequestActivity(
            String workspace,
            String repoSlug,
            String pullRequestId
    ) throws IOException {
        return remoteClient.getPullRequestActivity(workspace, repoSlug, pullRequestId);
    }

    @Override
    public Object approvePullRequest(
            String workspace,
            String repoSlug,
            String pullRequestId
    ) throws IOException {
        return remoteClient.approvePullRequest(workspace, repoSlug, pullRequestId);
    }

    @Override
    public Object unapprovePullRequest(
            String workspace,
            String repoSlug,
            String pullRequestId
    ) throws IOException {
        return remoteClient.unapprovePullRequest(workspace, repoSlug, pullRequestId);
    }

    @Override
    public Object declinePullRequest(
            String workspace,
            String repoSlug,
            String pullRequestId,
            String message
    ) throws IOException {
        return remoteClient.declinePullRequest(workspace, repoSlug, pullRequestId, message);
    }

    @Override
    public Object mergePullRequest(
            String workspace,
            String repoSlug,
            String pullRequestId,
            String message,
            String strategy
    ) throws IOException {
        return remoteClient.mergePullRequest(workspace, repoSlug, pullRequestId, message, strategy);
    }

    @Override
    public Object getPullRequestComments(
            String workspace,
            String repoSlug,
            String pullRequestId
    ) throws IOException {
        return remoteClient.getPullRequestComments(workspace, repoSlug, pullRequestId);
    }

    @Override
    public String getPullRequestDiff(
            String workspace,
            String repoSlug,
            String pullRequestId
    ) throws IOException {
        return remoteClient.getPullRequestDiff(workspace, repoSlug, pullRequestId);
    }

    @Override
    public Object getPullRequestCommits(
            String workspace,
            String repoSlug,
            String pullRequestId
    ) throws IOException {
        return remoteClient.getPullRequestCommits(workspace, repoSlug, pullRequestId);
    }

    @Override
    public Map<String, Object> getBranchingModel(String workspace, String repoSlug) throws IOException {
        return remoteClient.getBranchingModel(workspace, repoSlug);
    }

    @Override
    public Map<String, Object> getBranchingModelSettings(String workspace, String repoSlug) throws IOException {
        return remoteClient.getBranchingModelSettings(workspace, repoSlug);
    }

    @Override
    public Map<String, Object> updateBranchingModelSettings(
            String workspace,
            String repoSlug,
            Map<String, Object> development,
            Map<String, Object> production,
            List<Map<String, Object>> branchTypes
    ) throws IOException {
        return remoteClient.updateBranchingModelSettings(
                workspace, repoSlug, development, production, branchTypes);
    }

    @Override
    public String getBranchFileContent(
            String workspace,
            String repoSlug,
            String branch,
            String filePath
    ) throws IOException {
        if (!isLocalRef(workspace, repoSlug, branch)) {
            return remoteClient.getBranchFileContent(workspace, repoSlug, branch, filePath);
        }
        try {
            Path file = resolveExistingPath(filePath);
            if (!Files.isRegularFile(file)) {
                throw new IOException("Repository path is not a file: " + filePath);
            }
            return Files.readString(file, StandardCharsets.UTF_8);
        } catch (NoSuchFileException absentFromSnapshot) {
            return remoteClient.getBranchFileContent(
                    workspace, repoSlug, targetRevision, filePath);
        }
    }

    @Override
    public String getRootDirectory(
            String workspace,
            String repoSlug,
            String branch
    ) throws IOException {
        if (!isLocalRef(workspace, repoSlug, branch)) {
            return remoteClient.getRootDirectory(workspace, repoSlug, branch);
        }
        return readDirectory("");
    }

    @Override
    public String getDirectoryByPath(
            String workspace,
            String repoSlug,
            String branch,
            String dirPath
    ) throws IOException {
        if (!isLocalRef(workspace, repoSlug, branch)) {
            return remoteClient.getDirectoryByPath(workspace, repoSlug, branch, dirPath);
        }
        try {
            return readDirectory(dirPath);
        } catch (NoSuchFileException absentFromSnapshot) {
            return remoteClient.getDirectoryByPath(
                    workspace, repoSlug, targetRevision, dirPath);
        }
    }
}
