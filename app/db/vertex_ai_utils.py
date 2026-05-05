"""
Utility functions for Vertex AI Cloud SQL integration.
Provides health checks and diagnostics for the integration.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config.settings import settings

logger = logging.getLogger(__name__)


class VertexAIIntegrationStatus:
    """
    Check and report on Vertex AI Cloud SQL integration status.
    """
    
    @staticmethod
    def check_extension_installed(session: Session) -> bool:
        """
        Check if google_ml_integration extension is installed.
        
        Returns:
            True if extension is installed, False otherwise
        """
        try:
            stmt = text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension 
                    WHERE extname = 'google_ml_integration'
                ) as installed
            """)
            result = session.execute(stmt).scalar()
            return bool(result)
        except Exception as e:
            logger.error(f"[Vertex AI] Failed to check extension status: {e}")
            return False
    
    @staticmethod
    def check_embedding_function_exists(session: Session) -> bool:
        """
        Check if aidaan.get_embedding function exists.
        
        Returns:
            True if function exists, False otherwise
        """
        try:
            stmt = text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.routines
                    WHERE routine_schema = 'aidaan'
                    AND routine_name = 'get_embedding'
                ) as exists
            """)
            result = session.execute(stmt).scalar()
            return bool(result)
        except Exception as e:
            logger.error(f"[Vertex AI] Failed to check embedding function: {e}")
            return False
    
    @staticmethod
    def check_vector_column_exists(session: Session) -> bool:
        """
        Check if content_vector column exists in messages table.
        
        Returns:
            True if column exists, False otherwise
        """
        try:
            stmt = text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = :schema
                    AND table_name = 'messages'
                    AND column_name = 'content_vector'
                ) as exists
            """)
            result = session.execute(stmt, {"schema": settings.DB_AIDAAN_SCHEMA}).scalar()
            return bool(result)
        except Exception as e:
            logger.error(f"[Vertex AI] Failed to check vector column: {e}")
            return False
    
    @staticmethod
    def test_embedding_generation(session: Session) -> Optional[float]:
        """
        Test embedding generation with a sample text.
        
        Returns:
            Latency in milliseconds if successful, None if failed
        """
        import time
        
        try:
            start = time.monotonic()
            stmt = text("SELECT aidaan.get_embedding('test message')")
            session.execute(stmt).scalar()
            elapsed_ms = (time.monotonic() - start) * 1000
            return elapsed_ms
        except Exception as e:
            logger.error(f"[Vertex AI] Embedding generation test failed: {e}")
            return None
    
    @staticmethod
    def get_embedding_coverage(session: Session) -> Dict[str, Any]:
        """
        Get statistics on embedding coverage in messages table.
        
        Returns:
            Dictionary with total, with_embeddings, without_embeddings, coverage_pct
        """
        try:
            stmt = text(f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(content_vector) as with_embeddings,
                    COUNT(*) - COUNT(content_vector) as without_embeddings,
                    ROUND(100.0 * COUNT(content_vector) / NULLIF(COUNT(*), 0), 2) as coverage_pct
                FROM {settings.DB_AIDAAN_SCHEMA}.messages
                WHERE content IS NOT NULL
            """)
            result = session.execute(stmt).mappings().first()
            return dict(result) if result else {
                "total": 0,
                "with_embeddings": 0,
                "without_embeddings": 0,
                "coverage_pct": 0.0
            }
        except Exception as e:
            logger.error(f"[Vertex AI] Failed to get embedding coverage: {e}")
            return {
                "total": 0,
                "with_embeddings": 0,
                "without_embeddings": 0,
                "coverage_pct": 0.0,
                "error": str(e)
            }
    
    @staticmethod
    def get_integration_status(session: Session) -> Dict[str, Any]:
        """
        Get comprehensive Vertex AI integration status.
        
        Returns:
            Dictionary with all status checks and diagnostics
        """
        status = {
            "extension_installed": VertexAIIntegrationStatus.check_extension_installed(session),
            "embedding_function_exists": VertexAIIntegrationStatus.check_embedding_function_exists(session),
            "vector_column_exists": VertexAIIntegrationStatus.check_vector_column_exists(session),
        }
        
        # Only test embedding if prerequisites are met
        if status["extension_installed"] and status["embedding_function_exists"]:
            status["embedding_test_latency_ms"] = VertexAIIntegrationStatus.test_embedding_generation(session)
            status["embedding_test_passed"] = status["embedding_test_latency_ms"] is not None
        else:
            status["embedding_test_latency_ms"] = None
            status["embedding_test_passed"] = False
        
        # Get coverage stats if vector column exists
        if status["vector_column_exists"]:
            coverage = VertexAIIntegrationStatus.get_embedding_coverage(session)
            status.update(coverage)
        else:
            status.update({
                "total": 0,
                "with_embeddings": 0,
                "without_embeddings": 0,
                "coverage_pct": 0.0
            })
        
        # Overall status
        status["integration_ready"] = (
            status["extension_installed"] and
            status["embedding_function_exists"] and
            status["vector_column_exists"] and
            status["embedding_test_passed"]
        )
        
        # Recommendations
        recommendations = []
        if not status["extension_installed"]:
            recommendations.append("Install google_ml_integration extension: CREATE EXTENSION google_ml_integration CASCADE;")
        if not status["embedding_function_exists"]:
            recommendations.append("Run migration: migrations/enable_vertex_integration.sql")
        if not status["vector_column_exists"]:
            recommendations.append("Run migration to add content_vector column")
        if status.get("coverage_pct", 0) < 95:
            recommendations.append(f"Run backfill_embeddings.py to improve coverage (current: {status.get('coverage_pct', 0)}%)")
        
        status["recommendations"] = recommendations
        
        return status
    
    @staticmethod
    def log_integration_status(session: Session) -> None:
        """
        Log comprehensive integration status for diagnostics.
        """
        status = VertexAIIntegrationStatus.get_integration_status(session)
        
        if status["integration_ready"]:
            logger.info(
                f"[Vertex AI] Integration READY | "
                f"coverage={status.get('coverage_pct', 0)}% | "
                f"latency={status.get('embedding_test_latency_ms', 0):.1f}ms"
            )
        else:
            logger.warning(
                f"[Vertex AI] Integration NOT READY | "
                f"extension={status['extension_installed']} | "
                f"function={status['embedding_function_exists']} | "
                f"column={status['vector_column_exists']} | "
                f"test={status['embedding_test_passed']}"
            )
            
            if status["recommendations"]:
                logger.warning(f"[Vertex AI] Recommendations: {'; '.join(status['recommendations'])}")


def check_vertex_ai_integration(session: Session) -> bool:
    """
    Quick check if Vertex AI integration is ready.
    
    Returns:
        True if integration is ready, False otherwise
    """
    try:
        status = VertexAIIntegrationStatus.get_integration_status(session)
        return status["integration_ready"]
    except Exception as e:
        logger.error(f"[Vertex AI] Integration check failed: {e}")
        return False
