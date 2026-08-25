"""
Unit tests for utils.mcp_config — MCPConfigBuilder.
"""
import os
import sys
import pytest
from unittest.mock import patch
from utils.mcp_config import MCPConfigBuilder


class TestBuildConfig:

    def test_basic_config(self):
        result = MCPConfigBuilder.build_config("/path/to/server.jar")
        assert "mcpServers" in result
        assert "codecrow-vcs-mcp" in result["mcpServers"]
        cfg = result["mcpServers"]["codecrow-vcs-mcp"]
        assert cfg["command"] == "java"
        assert "-jar" in cfg["args"]
        assert "/path/to/server.jar" in cfg["args"]

    def test_jvm_props(self):
        props = {"project.id": "123", "workspace": "myws"}
        result = MCPConfigBuilder.build_config("/server.jar", jvm_props=props)
        args = result["mcpServers"]["codecrow-vcs-mcp"]["args"]
        assert "-Dproject.id=123" in args
        assert "-Dworkspace=myws" in args

    def test_jvm_props_sanitize_newlines(self):
        props = {"key": "line1\nline2"}
        result = MCPConfigBuilder.build_config("/server.jar", jvm_props=props)
        args = result["mcpServers"]["codecrow-vcs-mcp"]["args"]
        assert any("line1 line2" in a for a in args)

    @patch.dict(os.environ, {"MCP_DEBUG_PORT": "5005"})
    def test_debug_port(self):
        result = MCPConfigBuilder.build_config("/server.jar")
        args = result["mcpServers"]["codecrow-vcs-mcp"]["args"]
        assert any("5005" in a for a in args)
        assert any("agentlib" in a for a in args)

    @patch("os.path.exists", return_value=True)
    def test_platform_mcp_included(self, mock_exists):
        result = MCPConfigBuilder.build_config(
            "/vcs.jar",
            include_platform_mcp=True,
            platform_mcp_jar_path="/platform.jar",
            platform_jvm_props={"key": "val"},
        )
        assert "codecrow-platform-mcp" in result["mcpServers"]
        pcfg = result["mcpServers"]["codecrow-platform-mcp"]
        assert "-jar" in pcfg["args"]
        assert "/platform.jar" in pcfg["args"]

    @patch("os.path.exists", return_value=False)
    def test_platform_mcp_not_included_if_jar_missing(self, mock_exists):
        result = MCPConfigBuilder.build_config(
            "/vcs.jar",
            include_platform_mcp=True,
            platform_mcp_jar_path="/missing.jar",
        )
        assert "codecrow-platform-mcp" not in result["mcpServers"]

    def test_no_platform_by_default(self):
        result = MCPConfigBuilder.build_config("/vcs.jar")
        assert "codecrow-platform-mcp" not in result["mcpServers"]

    @patch.dict(
        os.environ,
        {
            "RAG_API_URL": "http://rag.test:8001",
            "SERVICE_SECRET": "test-secret",
        },
    )
    def test_rag_mcp_uses_request_binding_and_current_python(self):
        result = MCPConfigBuilder.build_config(
            "/vcs.jar",
            rag_mcp_context={
                "workspace": "tenant-workspace",
                "project": "project-namespace",
                "branch": "main",
                "revision": "abc123",
                "manifest": "manifest-sha",
                "collection_target": "collection",
            },
        )

        config = result["mcpServers"]["codecrow-rag-mcp"]
        assert config["command"] == sys.executable
        assert config["args"] == ["-m", "service.rag.rag_mcp_server"]
        assert config["env"]["CODECROW_RAG_MCP_WORKSPACE"] == "tenant-workspace"
        assert config["env"]["CODECROW_RAG_MCP_REVISION"] == "abc123"
        assert config["env"]["RAG_API_URL"] == "http://rag.test:8001"
        assert config["env"]["SERVICE_SECRET"] == "test-secret"


class TestBuildJvmProps:

    def test_basic_props(self):
        result = MCPConfigBuilder.build_jvm_props(
            project_id=1, pull_request_id=42,
            workspace="ws", repo_slug="repo",
        )
        assert result["project.id"] == "1"
        assert result["pullRequest.id"] == "42"
        assert result["workspace"] == "ws"
        assert result["repo.slug"] == "repo"

    def test_access_token(self):
        result = MCPConfigBuilder.build_jvm_props(
            project_id=1, pull_request_id=1,
            workspace="ws", repo_slug="r",
            access_token="tok123",
        )
        assert result["accessToken"] == "tok123"
        assert "oAuthClient" not in result

    def test_oauth(self):
        result = MCPConfigBuilder.build_jvm_props(
            project_id=1, pull_request_id=1,
            workspace="ws", repo_slug="r",
            oAuthClient="client", oAuthSecret="secret",
        )
        assert result["oAuthClient"] == "client"
        assert result["oAuthSecret"] == "secret"

    def test_explicit_token_limit_property_is_emitted(self):
        result = MCPConfigBuilder.build_jvm_props(
            project_id=1, pull_request_id=1,
            workspace="ws", repo_slug="r",
            max_allowed_tokens=40_000,
        )
        assert result["max.allowed.tokens"] == "40000"

    def test_vcs_provider(self):
        result = MCPConfigBuilder.build_jvm_props(
            project_id=1, pull_request_id=1,
            workspace="ws", repo_slug="r",
            vcs_provider="github",
        )
        assert result["vcs.provider"] == "github"

    def test_local_repository_binding(self):
        result = MCPConfigBuilder.build_jvm_props(
            project_id=1,
            pull_request_id=42,
            workspace="ws",
            repo_slug="repo",
            local_repo_path="/tmp/codecrow-pr-review-1",
            local_repo_target_branch="main",
            local_repo_revision="abc123",
        )

        assert result["local.repo.path"] == "/tmp/codecrow-pr-review-1"
        assert result["local.repo.targetBranch"] == "main"
        assert result["local.repo.revision"] == "abc123"

    def test_vcs_base_url(self):
        result = MCPConfigBuilder.build_jvm_props(
            project_id=1, pull_request_id=1,
            workspace="ws", repo_slug="r",
            vcs_provider="gitlab",
            vcs_base_url="https://gitlab.example.com",
        )
        assert result["vcs.baseUrl"] == "https://gitlab.example.com"

    def test_none_values_excluded(self):
        result = MCPConfigBuilder.build_jvm_props(
            project_id=None, pull_request_id=None,
            workspace=None, repo_slug=None,
        )
        assert "project.id" not in result
        assert "workspace" not in result
